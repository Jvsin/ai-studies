using System;
using Microsoft.SqlServer.Server;

[Serializable]
[SqlUserDefinedAggregate(Format.Native)]
public struct GEOAVE
{
    private int counter; // field to count the not null rows
    private double iloczyn;
    public void Init()
    {
        counter = 0;
        iloczyn = 1;// initialization
    }
    public void Accumulate(double? Value)
    {
        if (Value != null) // count just the rows that are not equal to NULL
        {
            counter++;
            iloczyn = iloczyn * (double)Value;
        }

    }
    public void Merge(GEOAVE Group)
    {
        this.iloczyn *= Group.iloczyn;
        this.counter += Group.counter;// when merge is needed the counter of other groups should be added
    }
    public double Terminate()
    {
        iloczyn = Math.Pow(iloczyn, 1.0 / counter);
        return this.iloczyn; //returning the results
    }

    public void Accumulate(double? value)
    {
        if (!value.isNull)
        {
            counter = counter + value.ToString().Length;
        }
    }

     
}