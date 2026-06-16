IF EXISTS (SELECT * FROM sys.types WHERE name = 'GeoPoint')
    DROP TYPE dbo.GeoPoint;
GO 

IF EXISTS (SELECT * FROM sys.assemblies WHERE name = 'GeoPoint_a')
    DROP ASSEMBLY GeoPoint_a;
GO 

CREATE ASSEMBLY GeoPoint_a 
AUTHORIZATION dbo 
FROM 'D:\projekty-bazy\GeoPoint\GeoPoint\bin\Debug\GeoPoint.dll' -- tutaj należy ustawić poprawną ścieżkę
WITH PERMISSION_SET = SAFE 
GO 

-- Utworzenie nowego typu opierającego się na wgranym assembly
CREATE TYPE dbo.GeoPoint  
EXTERNAL NAME GeoPoint_a.GeoPoint;
GO


-- Deklaracja zmiennych
DECLARE @lodz GeoPoint = CAST('51.7592,19.4560' AS GeoPoint);
DECLARE @warszawa GeoPoint = CONVERT(GeoPoint, '52.2297,21.0122');
DECLARE @nowyJork GeoPoint = '40.7128,-74.0060';
DECLARE @tokio GeoPoint = '35.6762,139.6503';


-- Test 1: Odczyt formatu tekstowego (ToString):
SELECT @lodz.ToString() AS Lodz, @nowyJork.ToString() AS NowyJork;

-- Test 2: Dostęp do pojedynczych właściwości
SELECT 
    @warszawa.Latitude AS Wawa_Szerokosc, 
    @warszawa.Longitude AS Wawa_Dlugosc;

-- Test 3: Metoda CzyWPromieniu 
-- Odległość z Łodzi do Warszawy to ok. 118 km w linii prostej.
SELECT 
    @lodz.CzyWPromieniu(@warszawa, 150) AS CzyWawaBlizejNiz150km,
    @lodz.CzyWPromieniu(@warszawa, 100) AS CzyWawaBlizejNiz100km;

-- Test 4: Metoda AzymutDo
-- Azymut z Łodzi do Warszawy to 64 stopnie
SELECT @lodz.AzymutDo(@warszawa) AS Azymut_Na_Warszawe;

-- Test 5: Metoda Antypody
-- Po drugiej stronie kuli ziemskiej od Łodzi znajduje się ocean na południe od Nowej Zelandii.
SELECT @lodz.Antypody().ToString() AS Antypody_Lodzi;

-- Test 6: Metoda LinkDoGoogleMaps
SELECT @tokio.LinkDoGoogleMaps() AS Tokio_Google_Maps_URL;

-- Test 8: Metoda przybliżonej strefy czasowej
SELECT 
    @lodz.PrzyblizonaStrefaCzasowa() AS StrefaCzasowa_Lodz_UTC,
    @nowyJork.PrzyblizonaStrefaCzasowa() AS StrefaCzasowa_NowyJork_UTC,
    @tokio.PrzyblizonaStrefaCzasowa() AS StrefaCzasowa_Tokio_UTC;