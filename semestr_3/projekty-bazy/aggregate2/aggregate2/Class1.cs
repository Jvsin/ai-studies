using Microsoft.SqlServer.Server;
using System;
using System.Collections.Generic;
[Serializable]
[SqlUserDefinedAggregate(Format.UserDefined,
IsInvariantToDuplicates = false,
IsInvariantToOrder = false,
MaxByteSize = 8000, Name = "OdchylenieStandardowe")]
public struct OdchStd : IBinarySerialize
{
    public List<double> posr;
    private int licznik; // zmienna zliczająca liczbę wierszy nie mających wartości NULL
    private double suma; // zmienna przechowująca sumę pól
    private double temp; // zmienna pomocnicza
    public void Init()
    {
        posr = new List<double>();
        licznik = 0;
        suma = 0;
    }
    public void Accumulate(double? value)
    {
        if (value != null) // wyznaczanie sumy dla pól nie mających wartości NULL
        {
            licznik++;
            temp = (double)value;
            suma = suma + temp;
            posr.Add(temp);
        }
    }
    public void Merge(OdchStd Group)
    {
        this.suma += Group.suma;
        this.licznik += Group.licznik;
        this.temp += Group.temp;
        posr.AddRange(Group.posr);
        // kiedy obliczenia równoległe suma staje się sumą z wszystkich procesów
    }
    public double? Terminate() //SqlString
    {
        if (licznik <= 1)
        {
            return null;
        }
        else
        {
            return (double?)(this.temp); //zwrócenie ostatecznej wartości obliczeń
        }
    }
    public void Write(System.IO.BinaryWriter w)
    {
        temp = 0;
        if (licznik > 1)
        {
            double sr = suma / licznik;
            foreach (double d in posr)
            {
                temp = temp + Math.Pow((d - sr), 2);
            }
            temp = temp / (licznik - 1);
            temp = Math.Pow(temp, 0.5);
        }
        w.Write(temp);
        //w.Write(suma);
        w.Write(licznik);
    }
    public void Read(System.IO.BinaryReader r)
    {
        temp = r.ReadDouble();
        //suma = r.ReadDouble();
        licznik = r.ReadInt32();
    }
}


[Serializable]
[SqlUserDefinedAggregate(Format.Native)]
public struct Kowariancja
{
    //public List<double> posr;
    private int licznik; // zmienna zliczająca liczbę wierszy nie mających wartości NULL
    private double sumaX; // zmienna przechowująca sumę pól
    private double sumaY; // zmienna przechowująca sumę pól
    private double sumaXY; // zmienna przechowująca sumę pól
    private double temp; // zmienna pomocnicza
    public void Init()
    {
        //posr = new List<double>();
        licznik = 0;
        sumaX = 0;
        sumaY = 0;
        sumaXY = 0;
    }
    public void Accumulate(double? X, double? Y)
    {
        if (X != null && Y != null) // wyznaczanie sumy dla pól nie mających wartości NULL
        {
            licznik++;
            sumaX = sumaX + (double)X;
            sumaY = sumaY + (double)Y;
            sumaXY = sumaXY + (double) (X*Y);
            //posr.Add(temp);
        }
    }
    public void Merge(Kowariancja Group)
    {
        this.sumaX += Group.sumaX;
        this.sumaY += Group.sumaY;
        this.sumaXY += Group.sumaXY;
        this.licznik += Group.licznik;
        //this.temp += Group.temp;
        //posr.AddRange(Group.posr);
        // kiedy obliczenia równoległe suma staje się sumą z wszystkich procesów
    }
    public double? Terminate() //SqlString
    {
        if (licznik <= 1)
        {
            return null;
        }
        else
        {   
            return sumaXY / licznik - (sumaX * sumaY) / (licznik * licznik);
        }
    }
    //public void Write(System.IO.BinaryWriter w)
    //{
    //    temp = 0;


    //    temp = 0;
    //    if (licznik > 1)
    //    {
    //        double srX = sumaX / licznik;
    //        double srY = sumaY / licznik;
    //        double srXY = sumaXY / licznik;
    //        foreach (double d in posr)
    //        {
    //            temp = temp + Math.Pow((d - sr), 2);
    //        }
    //        temp = temp / (licznik - 1);
    //        temp = Math.Pow(temp, 0.5);
    //    }
    //    w.Write(temp);
    //    //w.Write(suma);
    //    w.Write(licznik);
    //}
    //public void Read(System.IO.BinaryReader r)
    //{
    //    temp = r.ReadDouble();
    //    //suma = r.ReadDouble();
    //    licznik = r.ReadInt32();
    //}
}


[Serializable]
[SqlUserDefinedAggregate(Format.UserDefined, MaxByteSize = 8000)]
public struct Korelacja : IBinarySerialize
{
    public List<double> posrX;
    public List<double> posrY;
    private int licznik; // zmienna zliczająca liczbę wierszy nie mających wartości NULL
    private double sumaX; // zmienna przechowująca sumę pól
    private double sumaY; // zmienna przechowująca sumę pól
    private double sumaXY; // zmienna przechowująca sumę pól
    private double tempX; // zmienna pomocnicza
    private double tempY; // zmienna pomocnicza
    public void Init()
    {
        posrX = new List<double>();
        posrY = new List<double>();
        licznik = 0;
        sumaX = 0;
        sumaY = 0;
        sumaXY = 0;
        tempX = 0; tempY = 0;
    }
    public void Accumulate(double? X, double? Y)
    {
        if (X != null && Y != null) // wyznaczanie sumy dla pól nie mających wartości NULL
        {
            licznik++;
            sumaX = sumaX + (double)X;
            sumaY = sumaY + (double)Y;
            sumaXY = sumaXY + (double)(X * Y);
            this.posrX.Add((double) X);
            this.posrY.Add((double) Y);
        }
    }
    public void Merge(Korelacja Group)
    {
        this.sumaX += Group.sumaX;
        this.sumaY += Group.sumaY;
        this.sumaXY += Group.sumaXY;
        this.licznik += Group.licznik;
        this.posrX.AddRange(Group.posrX);
        this.posrY.AddRange(Group.posrY);
    }
    public double? Terminate() //SqlString
    {
        if (licznik == 0 || tempX == 0 || tempY == 0)
        {
            return null;
        }
        else
        {
            double covar = sumaXY / licznik - (sumaX * sumaY) / (licznik * licznik);
            return covar / (tempX * tempY);
        }
    }
    public void Write(System.IO.BinaryWriter w)
    {
        tempX = 0;
        tempY = 0;
        if (licznik > 0)
        {
            double srX = sumaX / licznik;
            foreach (double d in posrX)
            {
                tempX = tempX + Math.Pow((d - srX), 2);
            }
            tempX = tempX / (licznik);
            tempX = Math.Pow(tempX, 0.5);

            double srY = sumaY / licznik;
            foreach (double d in posrY)
            {
                tempY = tempY + Math.Pow((d - srY), 2);
            }
            tempY = tempY / (licznik);
            tempY = Math.Pow(tempY, 0.5);
        }
        w.Write(sumaX);
        w.Write(sumaY);
        w.Write(tempX);
        w.Write(tempY);
        w.Write(licznik);
    }
    public void Read(System.IO.BinaryReader r)
    {
        //    temp = r.ReadDouble();
        //    //suma = r.ReadDouble();
        //    licznik = r.ReadInt32();
    }
}