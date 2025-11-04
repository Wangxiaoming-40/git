using System;

public class ModbusParser
{
    public static string ParseModbus(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
            return "输入为空或无效。";

        try
        {
            string[] bytes = raw.Split(' ');
            if (bytes.Length < 18)
                return "报文不完整。";

            // 找到0x0A（10）之后的10个字节
            int index = Array.FindIndex(bytes, b => b == "0A");
            if (index == -1 || index + 10 >= bytes.Length)
                return "数据段不完整。";

            // 解析5个中间轴承温度
            string result = "";
            for (int i = 0; i < 5; i++)
            {
                string high = bytes[index + 1 + i * 2];
                string low = bytes[index + 2 + i * 2];
                int value = Convert.ToInt32(high + low, 16);
                string binary = Convert.ToString(value, 2).PadLeft(16,'0');
                double temp = value / 10.0;
                result += $"中间轴承{i + 1}的温度：{temp:F1}℃，二进制显示为{binary}\n";
            }
            return result.TrimEnd();
        }
        catch
        {
            return "解析出错。";
        }
    }

    public static void Main()
    {
        string test1 = "06 33 00 00 00 0D 01 03 0A 00 BE 00 CE 00 C1 00 B5 00 00";
        string test2 = null;
        string test3 = "";
        string test4 = "00 0D 01 03 0A 00 BE 00 CE 00 C1";

        Console.WriteLine(ParseModbus(test1));
        Console.WriteLine(ParseModbus(test2));
        Console.WriteLine(ParseModbus(test3));
        Console.WriteLine(ParseModbus(test4));
    }
}
