import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import sys 
sys.stdout.reconfigure(encoding='utf-8')

def get_relaxed_gates(a, b):
    """
    Oblicza wyjścia wszystkich 16 bramek logicznych dla wejść a i b.
    Wejścia a, b są w zakresie [0, 1].
    Zwraca tensor o kształcie [batch, num_neurons, 16]
    """
    # Operatory pomocnicze dla czytelności (Probabilistyczne T-normy)
    # AND (iloczyn)
    op_and = a * b
    # OR (suma probabilistyczna: a + b - ab)
    op_or = a + b - op_and
    # XOR (a + b - 2ab)
    op_xor = a + b - 2 * op_and
    
    # Lista 16 operatorów zgodna z indeksami w artykule:
    ops = [
        torch.zeros_like(a),            # 0: FALSE
        op_and,                         # 1: AND
        a * (1 - b),                    # 2: A and NOT B
        a,                              # 3: A
        b * (1 - a),                    # 4: B and NOT A
        b,                              # 5: B
        op_xor,                         # 6: XOR
        op_or,                          # 7: OR
        1 - op_or,                      # 8: NOR (NOT OR)
        1 - op_xor,                     # 9: XNOR (NOT XOR)
        1 - b,                          # 10: NOT B
        1 - (b * (1 - a)),              # 11: A or NOT B (implikacja B->A)
        1 - a,                          # 12: NOT A
        1 - (a * (1 - b)),              # 13: B or NOT A (implikacja A->B)
        1 - op_and,                     # 14: NAND (NOT AND)
        torch.ones_like(a)              # 15: TRUE
    ]
    
    # Składamy wzdłuż ostatniego wymiaru
    return torch.stack(ops, dim=-1)

# --- CZĘŚĆ 2: Warstwa i Model DLGN ---

class LogicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
        # Inicjalizacja połączeń: Każdy neuron ma dokładnie 2 wejścia[cite: 27, 114].
        # Losujemy indeksy wejść raz i trzymamy je jako stałe (fixed topology)[cite: 46, 109].
        self.indices = torch.randint(0, input_dim, (2, output_dim))
        # Rejestrujemy jako buffer, żeby nie były parametrami do optymalizacji, ale zapisały się z modelem
        self.register_buffer('conn_indices', self.indices)
        
        # Parametry wagowe dla wyboru bramki (16 opcji na neuron)[cite: 56, 139].
        # Inicjalizacja z rozkładu normalnego[cite: 149].
        self.gate_weights = nn.Parameter(torch.randn(output_dim, 16))

    def forward(self, x, hard_mode=False):
        # x shape: [batch_size, input_dim]
        
        # 1. Pobierz wejścia dla neuronów zgodnie z ustalonymi połączeniami
        # shape a, b: [batch_size, output_dim]
        a = x[:, self.conn_indices[0]]
        b = x[:, self.conn_indices[1]]
        
        # 2. Oblicz wartości wszystkich 16 potencjalnych bramek
        # shape: [batch_size, output_dim, 16]
        gates_out = get_relaxed_gates(a, b)
        
        if hard_mode:
            # Tryb dyskretny (Inference): Wybierz bramkę o największej wadze[cite: 25, 152].
            # To symuluje działanie binarne.
            best_gate_idx = self.gate_weights.argmax(dim=-1) # [output_dim]
            
            # Wybieramy odpowiednie wyjścia
            # Używamy gather do wybrania odpowiedniej bramki dla każdego neuronu
            # indices shape musi pasować do gates_out: [batch, output_dim, 1]
            indices = best_gate_idx.view(1, -1, 1).expand(x.size(0), -1, -1)
            activation = gates_out.gather(2, indices).squeeze(-1)
            
        else:
            # Tryb różniczkowalny (Training): Softmax na wagach i średnia ważona[cite: 140].
            probs = F.softmax(self.gate_weights, dim=-1) # [output_dim, 16]
            
            # Ważona suma wyjść bramek: sum(probs * gates_out)
            # Broadcasting: [batch, out, 16] * [1, out, 16] -> sum over dim 2
            activation = (gates_out * probs.unsqueeze(0)).sum(dim=-1)
            
        return activation

class DiffLogicNet(nn.Module):
    def __init__(self, input_dim, num_layers, neurons_per_layer, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        self.neurons_per_layer = neurons_per_layer
        
        # Budowanie warstw "prostych" (stała liczba neuronów)[cite: 150].
        current_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(LogicLayer(current_dim, neurons_per_layer))
            current_dim = neurons_per_layer # Kolejna warstwa bierze wyjście poprzedniej
            
    def forward(self, x, hard_mode=False):
        # Flatten input: [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(x.size(0), -1)
        
        # Przejście przez warstwy logiczne
        for layer in self.layers:
            x = layer(x, hard_mode=hard_mode)
            
        # Agregacja wyjść (Summation voting)[cite: 142, 144].
        # Dzielimy neurony ostatniej warstwy na grupy dla każdej klasy.
        # Np. jeśli mamy 10 klas i 1000 neuronów, każda klasa sumuje 100 neuronów.
        
        neurons_per_class = self.neurons_per_layer // self.num_classes
        if neurons_per_class == 0:
            raise ValueError("Za mało neuronów w warstwie, by podzielić na klasy.")
            
        # Reshape do [batch, num_classes, neurons_per_class]
        outputs_reshaped = x.view(x.size(0), self.num_classes, -1)
        
        # Sumujemy aktywacje w grupach (dla klasyfikacji)
        class_scores = outputs_reshaped.sum(dim=2)
        
        # Normalizacja temperaturą (według artykułu tau pomaga w zbieżności) [cite: 145, 161]
        # Dla MNIST tau ~ 0.01 - 0.1
        tau = 0.1
        return class_scores / tau

# --- CZĘŚĆ 3: Uczenie i Ewaluacja ---

def train_and_evaluate():
    # Parametry zaciągnięte z artykułu:
    BATCH_SIZE = 100
    EPOCHS = 5
    LR = 0.01
    LAYERS = 4
    NEURONS = 2000 # Duża liczba neuronów jest potrzebna dla rzadkich sieci logicznych
    
    # Urządzenie
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")

    # Dane: MNIST
    # Ważne: Sieci logiczne operują na wartościach binarnych/z zakresu [0,1].
    # MNIST jest w skali szarości, więc normalizujemy go do [0,1]. 
    # Dla "twardego" inference obraz też powinien być zbinaryzowany.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.round(x)) # Binaryzacja wejścia (0 lub 1) [cite: 45]
    ])

    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True)
    
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False)

    # Inicjalizacja modelu
    model = DiffLogicNet(input_dim=784, 
                         num_layers=LAYERS, 
                         neurons_per_layer=NEURONS, 
                         num_classes=10).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Start treningu: {LAYERS} warstw, {NEURONS} neuronów/warstwę.")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass w trybie różniczkowalnym (hard_mode=False)
            output = model(data, hard_mode=False)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        soft_correct = 0
        hard_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in enumerate(test_loader):
                # Poprawka pętli - enumerate zwraca (idx, data)
                pass 
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # 1. Soft Accuracy (jak sieć neuronowa)
                soft_out = model(data, hard_mode=False)
                pred_soft = soft_out.argmax(dim=1, keepdim=True)
                soft_correct += pred_soft.eq(target.view_as(pred_soft)).sum().item()
                
                # 2. Hard Accuracy (Dyskretne bramki logiczne - cel ostateczny) 
                hard_out = model(data, hard_mode=True)
                pred_hard = hard_out.argmax(dim=1, keepdim=True)
                hard_correct += pred_hard.eq(target.view_as(pred_hard)).sum().item()
                
                total += target.size(0)

        epoch_time = time.time() - start_time
        print(f"Epoka {epoch}: "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Soft Acc: {100. * soft_correct / total:.2f}% | "
              f"Hard Acc (Discretized): {100. * hard_correct / total:.2f}% | "
              f"Czas: {epoch_time:.1f}s")

    print("\n--- Analiza rozkładu bramek w ostatniej warstwie ---")
    # Sprawdzamy, jakich operatorów nauczyła się sieć
    last_layer_weights = model.layers[-1].gate_weights.detach().cpu()
    gate_indices = last_layer_weights.argmax(dim=1)
    
    # Nazwy zgodne z implementacją
    gate_names = ["FALSE", "AND", "A&!B", "A", "B&!A", "B", "XOR", "OR", 
                  "NOR", "XNOR", "!B", "A|!B", "!A", "B|!A", "NAND", "TRUE"]
    
    counts = torch.bincount(gate_indices, minlength=16)
    print("Liczba wystąpień poszczególnych bramek:")
    for i, count in enumerate(counts):
        if count > 0:
            print(f"{gate_names[i]}: {count.item()}")

if __name__ == '__main__':
    train_and_evaluate()