using System;
using System.Data.SqlTypes;
using Microsoft.SqlServer.Server;
using System.Text;
using System.Globalization;
using Microsoft.SqlServer.Types;

[Serializable]
[SqlUserDefinedType(Format.Native, ValidationMethodName = "SprawdzPunkt")]
public struct Punkt : INullable
{
    private bool is_Null;
    private double _x;
    private double _y;
    public bool IsNull
    {
        get
        { return (is_Null); }
    }
    public static Punkt Null
    {
        get
        {
            Punkt pt = new Punkt();
            pt.is_Null = true;
            return pt;
        }
    }
    public override string ToString()
    {
        if (this.IsNull)
            return "NULL";
        else
        {
            StringBuilder builder = new StringBuilder();
            builder.Append(String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _x));
            builder.Append(",");
            builder.Append(String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _y));
            return builder.ToString();
        }
    }
    [SqlMethod(OnNullCall = false)]
    public static Punkt Parse(SqlString s)
    {
        if (s.IsNull)
            return Null;
        Punkt pt = new Punkt();
        string[] xy = s.Value.Split(",".ToCharArray());
        pt.X = double.Parse(xy[0], CultureInfo.InvariantCulture.NumberFormat);
        pt.Y = double.Parse(xy[1], CultureInfo.InvariantCulture.NumberFormat);
        if (!pt.SprawdzPunkt())
            throw new ArgumentException("Invalid XY coordinate values.");
        return pt;
    }
    // współrzędne X i Y są ustawiane jako właściwościtypu.
    public double X
    {
        get
        { return this._x; }
        set
        {
            double temp = _x;
            _x = value;
            if (!SprawdzPunkt())
            {
                _x = temp;
                throw new ArgumentException("Zła współrzędna X.");
            }
        }
    }
    public double Y
    {
        get
        { return this._y; }
        set
        {
            double temp = _y;
            _y = value;
            if (!SprawdzPunkt())
            {
                _y = temp;
                throw new ArgumentException("Zła współrzędna Y.");
            }
        }
    }
    // metoda walidująca współrzędne X Y
    private bool SprawdzPunkt()
    {
        if ((_x >= 0) && (_y >= 0))
        { return true; }
        else
        { return false; }
    }
    // Odległość od 0,0.
    [SqlMethod(OnNullCall = false)]
    public Double Odleglosc()
    { return OdlegloscOdXY(0, 0); }
    // Odległość od wskazanego punktu
    [SqlMethod(OnNullCall = false)]
    public Double OdlegloscOd(Punkt pFrom)
    { return OdlegloscOdXY(pFrom.X, pFrom.Y); }
    // Odległość od wskazanego punktu.
    [SqlMethod(OnNullCall = false)]
    public Double OdlegloscOdXY(double iX, double iY)
    {
        return Math.Sqrt(Math.Pow(iX - _x, 2.0) + Math.Pow(iY - _y, 2.0));
    }
    //[SqlMethod(OnNullCall = false)]
    //public SqlChars toSpatial()
    //{
    //    SqlGeometry sqlg =
    //    SqlGeometry.STGeomFromText(
    //    new SqlChars(
    //        new SqlString("POINT(" +
    //         String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _x) + " " +
    //        String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _y)) + ")"), 0);
    //        return sqlg.STAsText();

    //    }
    //[SqlMethod(OnNullCall = false)]
    //public SqlGeometry ToSpatial()
    //{
    //    SqlGeometry sqlg =
    //SqlGeometry.STGeomFromText(
    //new SqlChars(
    //    new SqlString("POINT(" +
    //     String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _x) + " " +
    //    String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _y)) + ")"), 0);
    //    return sqlg.STBuffer(.1);

    //}

    //[SqlMethod(OnNullCall = false)]
    //public SqlGeometry ToSpatialLine()
    //{
    //    SqlGeometry sqlg =
    //SqlGeometry.STGeomFromText(
    //new SqlChars(
    //    new SqlString("LINESTRING(0 0," +
    //     String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _x) + " " +
    //    String.Format(CultureInfo.GetCultureInfo("en-US"), "{0,0}", _y)) + ")"), 0);
    //    return sqlg;

    //}
}