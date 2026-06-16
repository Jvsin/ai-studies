using Microsoft.SqlServer.Server;
using System;
using System.Collections;
using System.Data.SqlClient;
using System.Data.SqlTypes;
using System.Globalization;

public class wynik
{
    [SqlFunction]
    public static double? reszta(double? a, double? b)
    {
        double? wyn = a % b;
        return wyn;
    }

    [SqlProcedure]
    public static void resztaProcedure(double? a, double? b, out double? wyn)
    {
        wyn = a % b;
    }

    [SqlProcedure]
    public static void wiersze(String tabela, out SqlInt32 ile)
    {
        ile = 0;
        SqlConnection conn = new SqlConnection("context connection=true");

        conn.Open();

        SqlCommand zap = new SqlCommand("SELECT count(*) FROM " + tabela, conn);

        SqlDataReader zapReader = zap.ExecuteReader();

        while (zapReader.Read())
        {
            ile = zapReader.GetInt32(0);
        }
        conn.Close();
    }

    [SqlFunction(DataAccess = DataAccessKind.Read)]
    public static int wierszeFunkcja(String tabela)
    {
        int ile = 0;
        try
        {
            SqlConnection conn = new SqlConnection("context connection=true");

            conn.Open();

            SqlCommand zap = new SqlCommand("SELECT count(*) FROM " + tabela, conn);

            SqlDataReader zapReader = zap.ExecuteReader();

            while (zapReader.Read())
            {
                ile = zapReader.GetInt32(0);
            }
            conn.Close();
        }
        catch (Exception e)
        {
            ile = -5;
        }
        return ile;
    }

    private class ZapResult
    {
        public SqlInt32 idosoby;
        public SqlString nazwisko;
        public SqlSingle wzrost;

        public ZapResult(SqlInt32 IdOsoby, SqlString Nazwisko, SqlSingle Wzrost)
        {
            idosoby = IdOsoby;
            nazwisko = Nazwisko;
            wzrost = Wzrost;
        }
    }

    [SqlFunction(DataAccess = DataAccessKind.Read, FillRowMethodName = "Pobierz")]
    public static IEnumerable wysocy(double mini)
    {
        ArrayList resultCollection = new ArrayList();
        SqlConnection conn = new SqlConnection("context connection=true");
        conn.Open();
        SqlCommand zap = new SqlCommand("SELECT IdOsoby, Nazwisko, Wzrost FROM Osoby WHERE Wzrost > " + //mini, conn);
            String.Format(CultureInfo.GetCultureInfo("en-US"),"{0,0}", mini), conn);
        SqlDataReader zapReader = zap.ExecuteReader();
        while (zapReader.Read())
        {
            resultCollection.Add(new ZapResult(zapReader.GetSqlInt32(0),
            zapReader.GetSqlString(1), 
            zapReader.GetSqlSingle(2)));
        }
        return resultCollection;
    }

    public static void Pobierz(object zapResultObj, out SqlInt32 idosoby, out SqlString nazwisko, out SqlSingle wzrost)
    {
        ZapResult result = (ZapResult)zapResultObj;
        idosoby = result.idosoby;
        nazwisko = result.nazwisko;
        wzrost = result.wzrost;
    }
}


