using System;
using System.IO;
using System.Collections;
using System.Data.SqlTypes;
using System.Text;
using System.Globalization;
using Microsoft.SqlServer.Server;
using Microsoft.SqlServer.Types;

[Serializable]
[SqlUserDefinedType(Format.UserDefined,
     IsByteOrdered = true, ValidationMethodName = "ValidatePoint", MaxByteSize = 2000)]
public struct nPoint : INullable, IBinarySerialize
{
    private bool is_Null;
    private ArrayList coordinates;
    // konstruktor
    public nPoint(SqlString s)
    {
        is_Null = false;
        coordinates = new ArrayList();
        CultureInfo ci = new CultureInfo("en-US");
        double dblVal;
        char[] separator = { ',' };
        string[] _wspolrzedne = s.Value.Split(separator);
        for (int i = 0; i < _wspolrzedne.Length; i++)
        {
            try
            {
                dblVal = Double.Parse(_wspolrzedne[i], ci);
                coordinates.Add(dblVal);
            }
            catch (Exception ex)
            {
                throw new ArgumentException("Błąd konwersji:\n" + ex.Message);
            }
        }
    }
    public bool IsNull
    {
        get
        { return (is_Null); }
    }

    public static nPoint Null
    {
        get
        {
            nPoint pt = new nPoint();
            pt.is_Null = true;
            return pt;
        }
    }

    public int Size
    {
        get
        {
            return (coordinates.Count);
        }
    }
    public void Read(BinaryReader r)
    {
        int size = r.ReadInt32();
        coordinates = new ArrayList();
        for (int i = 0; i < size; i++)
            coordinates.Add(r.ReadDouble());
    }

    public void Write(BinaryWriter w)
    {
        w.Write(coordinates.Count);
        for (int i = 0; i < coordinates.Count; i++)
            w.Write((double)coordinates[i]);
    }
    public override string ToString()
    {
        if (this.IsNull)
            return "NULL";

        else
        {
            double dbl;
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < coordinates.Count; i++)
            {
                CultureInfo ci = new CultureInfo("en-US");
                dbl = (double)coordinates[i];
                builder.Append(dbl.ToString(ci));
                if (i != coordinates.Count - 1) builder.Append(",");
            }
            return builder.ToString();
        }
    }
    [SqlMethod(OnNullCall = false)]
    public static nPoint Parse(SqlString s)
    {
        if (s.IsNull || s == "")
            return Null;

        // Rozdziela przesłany łańcuch na poszczególne liczby typu double
        nPoint pt = new nPoint(s);

        // Przeprowadza walidację punktu (tutaj zbędna, bo każda liczba typu double
        // jest prawidłowa, ale istnienie jakiejś metody walidacyjnej jest wymagane)
        if (!pt.ValidatePoint())
            throw new ArgumentException("Nieprawidłowe wartości współrzędnych.");
        return pt;
    }
    [SqlMethod(OnNullCall = false)]
    public double Coordinate(SqlInt32 i)
    {
        int k = i.Value;
        if (k < 0 || k >= coordinates.Count)
            throw new ArgumentException("Nieprawidłowa wartość indeksu współrzędnej.");
        else
        {
            return ((Double)coordinates[k]);
        }
    }

    // Metoda walidująca współrzędne punktów
    private bool ValidatePoint()
    {
        return true;
    }
    // Metoda zwraca współrzędne jako string (get) i ustawia nowe z przesłanego stringa (set)
    public SqlString Coordinates
    {
        get
        { return ToString(); }
        set
        {
            coordinates = new ArrayList();
            CultureInfo ci = new CultureInfo("en-US");
            double dblVal;
            char[] separator = { ',' };
            string[] _wspolrzedne = value.Value.Split(separator);
            for (int i = 0; i < _wspolrzedne.Length; i++)
            {
                try
                {
                    dblVal = Double.Parse(_wspolrzedne[i], ci);
                    coordinates.Add(dblVal);
                }
                catch (Exception ex)
                {
                    throw new ArgumentException("Błąd konwersji:\n" + ex.Message);
                }
            }
        }
    }
    // Metoda przesuwa punkt o wektor, którego współrzędne przekazujemy parameterm jako SqlString
    [SqlMethod(OnNullCall = false)]
    public SqlString Przesun(SqlString wektor)
    {
        char[] separator = { ',' };
        string[] _wspolrzedne = wektor.Value.Split(separator);
        if (_wspolrzedne.Length > coordinates.Count)
            throw new ArgumentException("Zbyt duża liczba współrzędnych.");
        ArrayList _coordinates = new ArrayList();
        CultureInfo ci = new CultureInfo("en-US");
        double dblVal;
        for (int i = 0; i < _wspolrzedne.Length; i++)
        {
            try
            {
                dblVal = Double.Parse(_wspolrzedne[i], ci);
                _coordinates.Add(dblVal);
            }
            catch (Exception ex)
            {
                throw new ArgumentException("Błąd konwersji:\n" + ex.Message);
            }
        }
        for (int i = 0; i < _coordinates.Count; i++)
        { coordinates[i] = (double)coordinates[i] + (double)_coordinates[i]; }
        return new SqlString(ToString());
    }
    // Metoda obliczająca odległość od początku układu współrzędnych do danego punktu.
    [SqlMethod(OnNullCall = false)]
    public Double Distance()
    {
        double suma = 0;
        for (int i = 0; i < coordinates.Count; i++)
            suma += Math.Pow((double)coordinates[i], 2);
        return Math.Pow(suma, 0.5);
    }
    // Metoda obliczająca odległość od danego punktu do punktu przesłanego jako argument.
    [SqlMethod(OnNullCall = false)]
    public Double DistanceFrom(nPoint p)
    {
        if (Size != p.Size)
            throw new Exception("Niezgodne wymiary punktów!\n" +
"Nie można obliczyć odległości, jeśli punkt źródłowy " +
"i docelowy nie mają tego samego wymiaru.");
        double suma = 0;
        for (int i = 0; i < coordinates.Count; i++)
            suma += Math.Pow((double)coordinates[i] - (double)p.coordinates[i], 2);
        return Math.Pow(suma, 0.5);
    }
    // Metoda obliczająca odległość od danego punktu do punktu o współrzędnych przesłanych jako argument
    [SqlMethod(OnNullCall = false)]
    public Double DistanceFromCoordinates(SqlString s)
    {
        char[] separator = { ',' };
        string[] _wspolrzedne = s.Value.Split(separator);
        if (_wspolrzedne.Length > Size)
            throw new ArgumentException("Zbyt duża liczba współrzędnych.\n" + "Przesłana liczba współrzędnych musi być mniejsza lub " +
            "równa wymiarowi punktu.");
        CultureInfo ci = new CultureInfo("en-US");
        double dblVal, suma = 0;
        for (int i = 0; i < Size; i++)
        {
            if (i < _wspolrzedne.Length)
            {
                try
                {
                    dblVal = Double.Parse(_wspolrzedne[i], ci);
                    suma += Math.Pow((double)coordinates[i] - dblVal, 2);
                }
                catch (Exception ex)
                { throw new ArgumentException("Błąd konwersji:\n" + ex.Message); }
            }
            else
            {// przyjmujemy, że dalsze współrzędne (nie przesłane) wynoszą 0
                suma += Math.Pow((double)coordinates[i], 2);
            }
        }
        return Math.Pow(suma, 0.5);
    }
    [SqlMethod(OnNullCall = false)]
    public SqlChars toSpatial()
    {
        SqlGeometry sqlg =
            SqlGeometry.STGeomFromText(
            new SqlChars(
                new SqlString("LINESTRING(0 0, " +
                String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", this.Coordinate(0)) + " " +
                String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", this.Coordinate(1))) + ")"), 0);
        return sqlg.STAsText();
    }

    [SqlMethod(OnNullCall = false)]
    public SqlGeometry ToSpatial()
    {
        SqlGeometry sqlg =
            SqlGeometry.STGeomFromText(
            new SqlChars(
                new SqlString("LINESTRING(0 0, " +
                String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", this.Coordinate(0)) + " " +
                String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", this.Coordinate(1))) + ")"), 0);
        return sqlg;
    }
}
