using System;
using System.Data.SqlTypes;
using System.IO;
using System.Globalization;
using Microsoft.SqlServer.Server;

[Serializable]
[SqlUserDefinedType(Format.UserDefined, IsByteOrdered = true, ValidationMethodName = "ValidatePoint", MaxByteSize = 32)]
public struct GeoPoint : INullable, IBinarySerialize
{
    private bool is_Null;
    private double latitude; //szerokość geograficzna
    private double longitude; // długość geograficzna

    // konstruktor
    public GeoPoint(double lat, double lon)
    {
        latitude = lat;
        longitude = lon;
        is_Null = false;
    }

    public bool IsNull
    {
        get { return is_Null; }
    }

    // pusty konstruktor
    public static GeoPoint Null
    {
        get
        {
            GeoPoint pt = new GeoPoint();
            pt.is_Null = true;
            return pt;
        }
    }

    // gettery
    public double Latitude => latitude;
    public double Longitude => longitude;

    // funkcja toString
    public override string ToString()
    {
        if (this.IsNull) return "NULL";

        CultureInfo ci = new CultureInfo("en-US");
        return latitude.ToString(ci) + "," + longitude.ToString(ci);
    }

    [SqlMethod(OnNullCall = false)]
    public static GeoPoint Parse(SqlString s)
    {
        if (s.IsNull || s.Value == "")
            return Null;

        string[] parts = s.Value.Split(',');
        if (parts.Length != 2)
            throw new ArgumentException("Błędny format. Oczekiwano: 'szerokość,długość'.");

        CultureInfo ci = new CultureInfo("en-US");
        try
        {
            double lat = double.Parse(parts[0], ci);
            double lon = double.Parse(parts[1], ci);

            GeoPoint pt = new GeoPoint(lat, lon);

            if (!pt.ValidatePoint())
                throw new ArgumentException("Nieprawidłowe wartości. Szerokość [-90, 90], Długość [-180, 180].");

            return pt;
        }
        catch (Exception ex)
        {
            throw new ArgumentException("Błąd parsowania: " + ex.Message);
        }
    }

    // validate dla współrzędnych
    private bool ValidatePoint()
    {
        return (latitude >= -90.0 && latitude <= 90.0 && longitude >= -180.0 && longitude <= 180.0);
    }
    public void Read(BinaryReader r)
    {
        is_Null = r.ReadBoolean();
        if (!is_Null)
        {
            latitude = r.ReadDouble();
            longitude = r.ReadDouble();
        }
    }

    public void Write(BinaryWriter w)
    {
        w.Write(is_Null);
        if (!is_Null)
        {
            w.Write(latitude);
            w.Write(longitude);
        }
    }
    private double ToRadians(double angle) => Math.PI * angle / 180.0;
    private double ToDegrees(double angle) => angle * 180.0 / Math.PI;

    // Metoda CzyWPromieniu - sprawdza czy punkt znajduje się w promieniu drugiego punktu
    // wykorzystuje wzór Haversine'a który pozwala obliczyć najkrótszą trasę na sferze.
    [SqlMethod(OnNullCall = false)]
    public SqlBoolean CzyWPromieniu(GeoPoint srodek, SqlDouble promienWKilometrach)
    {
        double R = 6371.0; // promień ziemii
        double dLat = ToRadians(srodek.latitude - this.latitude);
        double dLon = ToRadians(srodek.longitude - this.longitude);

        double a = Math.Sin(dLat / 2) * Math.Sin(dLat / 2) +
                   Math.Cos(ToRadians(this.latitude)) * Math.Cos(ToRadians(srodek.latitude)) *
                   Math.Sin(dLon / 2) * Math.Sin(dLon / 2);

        double c = 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));
        double dystans = R * c;

        return new SqlBoolean(dystans <= promienWKilometrach.Value);
    }

    // AzymutDo - kierunek nawigacyjny do innego punktu w zakresie 0-360 stopni
    [SqlMethod(OnNullCall = false)]
    public SqlDouble AzymutDo(GeoPoint innyPunkt)
    {
        double dLon = ToRadians(innyPunkt.longitude - this.longitude);
        double y = Math.Sin(dLon) * Math.Cos(ToRadians(innyPunkt.latitude));
        double x = Math.Cos(ToRadians(this.latitude)) * Math.Sin(ToRadians(innyPunkt.latitude)) -
                   Math.Sin(ToRadians(this.latitude)) * Math.Cos(ToRadians(innyPunkt.latitude)) * Math.Cos(dLon);

        double brng = Math.Atan2(y, x);
        return new SqlDouble((ToDegrees(brng) + 360) % 360);
    }

    // Antypody - miejsce dokładnie po drugiej stronie Ziemi
    [SqlMethod(OnNullCall = false)]
    public GeoPoint Antypody()
    {
        double antLat = -latitude;
        double antLon = longitude <= 0 ? longitude + 180.0 : longitude - 180.0;

        return new GeoPoint(antLat, antLon);
    }

    // LinkDoGoogleMaps
    [SqlMethod(OnNullCall = false)]
    public SqlString LinkDoGoogleMaps()
    {
        CultureInfo ci = new CultureInfo("en-US");
        string url = $"https://www.google.com/maps?q={latitude.ToString(ci)},{longitude.ToString(ci)}";
        return new SqlString(url);
    }

    // PrzyblizonaStrefaCzasowa względem południka zerowego (UTC)
    // metoda czysto koncepcyjna, przyjęta że 1h to około 15 stopni.
    [SqlMethod(OnNullCall = false)]
    public SqlInt32 PrzyblizonaStrefaCzasowa()
    {
        int strefa = (int)Math.Round(longitude / 15.0);
        return new SqlInt32(strefa);
    }
}