DROP Type nPunkt  
DROP ASSEMBLY nPunkt_a
GO 
CREATE ASSEMBLY nPunkt_a 
AUTHORIZATION dbo 
FROM 'D:\projekty-bazy\n_point\n_point\bin\Debug\n_point.dll'
WITH PERMISSION_SET = SAFE 
GO 
CREATE TYPE dbo.nPunkt  
EXTERNAL NAME nPunkt_a.n_Point;
GO
DECLARE @a Punkt, @b Punkt, @c Punkt
SET @a = CAST('1, 4' AS Punkt)
SET @b = CONVERT(Punkt, '5, 7')
SET @c = '3, 6'

SELECT @a, @b, @c
SELECT @a.ToString(), @b.ToString(), @c.ToString()
SELECT @c.X
SET @c.X = 2
SELECT @c.X
SELECT @a.OdlegloscOdXY(4, 5), @a.OdlegloscOd(@b), @b.OdlegloscOd(@a), @a.Odleglosc()
SELECT @a.toSpatial(), @b.toSpatial(), @c.toSpatial()
SELECT @a.ToSpatial()
UNION ALL 
SELECT @b.ToSpatial()
UNION ALL
SELECT @c.ToSpatial()

SELECT @a.ToSpatialLine()
UNION ALL 
SELECT @b.ToSpatialLine()
UNION ALL
SELECT @c.ToSpatialLine()

SELECT name, clr_name
FROM sys.assemblies
WHERE name = 'microsoft.sqlserver.types';