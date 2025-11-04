using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

public class ShipSample
{
    public double SpeedKnots { get; set; }// 船速
    public double WindSpeed { get; set; }// 风速
    public double WindDirDeg { get; set; }// 风向
    public double HeadingDeg { get; set; }// 航向
    public double DisplacementTons { get; set; }// 吃水吨位
    public double Rpm { get; set; }// 转速
    public double SeaState { get; set; }// 海况
    public double FuelLph { get; set; } // 目标：燃油消耗
}
// CSV 加载器
public static class CsvLoader
{
    public static List<ShipSample> Load(string path)
    {
        var lines = File.ReadAllLines(path);
        if (lines.Length < 2) throw new Exception("CSV 为空或没有数据行。");

        var header = lines[0].Split(',');
        // 简化：用列名找下标
        int idxSpeed = Array.IndexOf(header, "speed_knots");
        int idxWindSpeed = Array.IndexOf(header, "wind_speed");
        int idxWindDir = Array.IndexOf(header, "wind_dir_deg");
        int idxHeading = Array.IndexOf(header, "heading_deg");
        int idxDisp = Array.IndexOf(header, "displacement_tons");
        int idxRpm = Array.IndexOf(header, "rpm");
        int idxSea = Array.IndexOf(header, "sea_state");
        int idxFuel = Array.IndexOf(header, "fuel_lph");

        int[] need = { idxSpeed, idxWindSpeed, idxWindDir, idxHeading, idxDisp, idxRpm, idxSea, idxFuel };
        if (need.Any(i => i < 0)) throw new Exception("缺少必要列名，请检查表头。");

        var list = new List<ShipSample>();
        for (int i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i])) continue;
            var cols = lines[i].Split(',');
            double D(int idx) => double.Parse(cols[idx].Trim(), System.Globalization.CultureInfo.InvariantCulture);

            list.Add(new ShipSample
            {
                SpeedKnots = D(idxSpeed),
                WindSpeed = D(idxWindSpeed),
                WindDirDeg = D(idxWindDir),
                HeadingDeg = D(idxHeading),
                DisplacementTons = D(idxDisp),
                Rpm = D(idxRpm),
                SeaState = D(idxSea),
                FuelLph = D(idxFuel),
            });
        }
        return list;
    }
}
// 特征工具
public static class FeatureUtils
{
    public static double DegToRad(double deg) => Math.PI * deg / 180.0;

    // 返回 (-180, 180] 的相对角：风向相对航向
    public static double RelativeAngle(double aDeg, double bDeg)
    {
        double d = (aDeg - bDeg + 180.0) % 360.0;
        if (d < 0) d += 360.0;
        d -= 180.0;
        return d;
    }
    // 返回角度的正弦和余弦
    public static (double sin, double cos) SinCosDeg(double deg)
    {
        double r = DegToRad(deg);
        return (Math.Sin(r), Math.Cos(r));
    }
}

// 简单的标准化器（z-score）
public class StandardScaler
{
    public double[] Mean { get; private set; }
    public double[] Std { get; private set; }

    public void Fit(double[][] X)
    {
        int n = X.Length;// 样本数
        int d = X[0].Length;// 特征数
        Mean = new double[d];
        Std = new double[d];

        for (int j = 0; j < d; j++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) s += X[i][j];
            Mean[j] = s / n;

            double v = 0;
            for (int i = 0; i < n; i++)
            {
                double z = X[i][j] - Mean[j];
                v += z * z;
            }
            Std[j] = Math.Sqrt(v / n) + 1e-8; // 防除零
        }
    }
    // 标准化
    public double[][] Transform(double[][] X)
    {
        int n = X.Length;
        int d = X[0].Length;
        var Y = new double[n][];
        for (int i = 0; i < n; i++)
        {
            Y[i] = new double[d];
            for (int j = 0; j < d; j++)
                Y[i][j] = (X[i][j] - Mean[j]) / Std[j];
        }
        return Y;
    }
    // 标准化单条样本
    public double[] TransformOne(double[] x)
    {
        var y = new double[x.Length];
        for (int j = 0; j < x.Length; j++)
            y[j] = (x[j] - Mean[j]) / Std[j];
        return y;
    }
}
// 特征构建
public static class FeatureBuilder
{
    // 返回：X(标准化前)，y，列名
    public static (double[][] X, double[] y, string[] cols) Build(List<ShipSample> data)
    {
        var cols = new List<string> {
            "speed_knots","wind_speed","displacement_tons","rpm","sea_state",
            "sin_rel_wind_heading","cos_rel_wind_heading"
        };

        double[][] X = new double[data.Count][];
        double[] y = new double[data.Count];

        for (int i = 0; i < data.Count; i++)
        {
            var s = data[i];
            double rel = FeatureUtils.RelativeAngle(s.WindDirDeg, s.HeadingDeg);
            var (sinRel, cosRel) = FeatureUtils.SinCosDeg(rel);

            X[i] = new double[]
            {
                s.SpeedKnots,
                s.WindSpeed,
                s.DisplacementTons,
                s.Rpm,
                s.SeaState,
                sinRel,
                cosRel
            };
            y[i] = s.FuelLph;
        }
        return (X, y, cols.ToArray());
    }
}
// 分桶器
public class Binner
{
    public double[] Edges { get; private set; }   // K+1 个边界
    public double[] Centers { get; private set; } // K 个桶心
    public string Strategy { get; private set; }  // "quantile" 或 "uniform"

    private Binner() { }
    // 拟合分桶器
    public static Binner Fit(double[] y, int K = 12, string strategy = "quantile")
    {
        if (K < 3) throw new ArgumentException("K 至少为 3。");
        var b = new Binner { Strategy = strategy };

        double[] edges;
        if (strategy == "quantile")
        {
            // 分位数边界
            edges = new double[K + 1];
            Array.Sort(y);
            for (int i = 0; i <= K; i++)
            {
                double q = (double)i / K;
                double pos = q * (y.Length - 1);
                int lo = (int)Math.Floor(pos);
                int hi = (int)Math.Ceiling(pos);
                if (lo == hi) edges[i] = y[lo];
                else edges[i] = y[lo] + (pos - lo) * (y[hi] - y[lo]);
            }
        }
        else if (strategy == "uniform")
        {
            double ymin = y.Min(), ymax = y.Max();
            edges = new double[K + 1];
            for (int i = 0; i <= K; i++)
                edges[i] = ymin + (ymax - ymin) * i / K;
        }
        else throw new ArgumentException("strategy 只能是 quantile 或 uniform");

        // 桶心
        var centers = new double[K];
        for (int i = 0; i < K; i++)
            centers[i] = 0.5 * (edges[i] + edges[i + 1]);

        // 去重保护
        edges = edges.Distinct().ToArray();
        if (edges.Length < K + 1)
        {
            centers = new double[edges.Length - 1];
            for (int i = 0; i < centers.Length; i++)
                centers[i] = 0.5 * (edges[i] + edges[i + 1]);
        }

        b.Edges = edges;
        b.Centers = centers;
        return b;
    }

    // 把 y 映射到桶索引
    public int[] Encode(double[] y)
    {
        int K = Edges.Length - 1;
        var idx = new int[y.Length];
        for (int i = 0; i < y.Length; i++)
        {
            // 找右侧边界位置-1
            int k = Array.BinarySearch(Edges, y[i]);
            if (k >= 0) k = Math.Min(k, K - 1);
            else
            {
                int ins = ~k;
                k = Math.Clamp(ins - 1, 0, K - 1);
            }
            idx[i] = k;
        }
        return idx;
    }
}
// 概率感知感知机
public class ProbabilisticPerceptron
{
    private readonly int _classes;
    private readonly int _dim; // 包含 bias 的维度
    private readonly double[][] _W; // K x (d+1)
    private readonly Random _rng;
    private readonly double _lr, _tau;
    private readonly bool _stochasticPred;
    private readonly double[] _centers;

    public ProbabilisticPerceptron(int nFeatures, int nClasses, double[] centers,
                                   double lr = 0.05, double tau = 0.4, bool stochasticPred = true, int seed = 42)
    {
        _classes = nClasses;// 类别数
        _dim = nFeatures + 1; // +1 做 bias
        _W = Enumerable.Range(0, nClasses).Select(_ => new double[_dim]).ToArray();// 初始化为 0
        _rng = new Random(seed);// 随机数生成器
        _lr = lr;// 学习率
        _tau = tau;// 温度参数
        _stochasticPred = stochasticPred;// 是否随机预测
        _centers = centers.ToArray();// 桶中心
    }
    // 添加偏置项
    private double[] AddBias(double[] x)
    {
        var xb = new double[_dim];
        Array.Copy(x, xb, x.Length);
        xb[_dim - 1] = 1.0;
        return xb;
    }
    // Softmax 函数
    private double[] Softmax(double[] z)
    {
        double max = z.Max();
        double[] ez = new double[z.Length];
        double sum = 0;
        for (int i = 0; i < z.Length; i++)
        {
            ez[i] = Math.Exp((z[i] - max) / _tau);
            sum += ez[i];
        }
        for (int i = 0; i < ez.Length; i++) ez[i] /= (sum + 1e-12);
        return ez;
    }
    // 按概率采样类别
    private int SampleByProb(double[] p)
    {
        double r = _rng.NextDouble();
        double c = 0;
        for (int i = 0; i < p.Length; i++)
        {
            c += p[i];
            if (r <= c) return i;
        }
        return p.Length - 1;
    }
    // 训练
    public void Fit(double[][] X, int[] yBin, int epochs = 15)
    {
        int n = X.Length;
        var idx = Enumerable.Range(0, n).ToArray();

        for (int ep = 0; ep < epochs; ep++)
        {
            // 打乱顺序
            idx = idx.OrderBy(_ => _rng.Next()).ToArray();

            foreach (int i in idx)
            {
                var xb = AddBias(X[i]);
                // margins
                var margins = new double[_classes];
                for (int k = 0; k < _classes; k++)
                {
                    double s = 0;
                    for (int j = 0; j < _dim; j++) s += _W[k][j] * xb[j];
                    margins[k] = s;
                }
                var probs = Softmax(margins);
                int yHat = _stochasticPred ? SampleByProb(probs) : Array.IndexOf(margins, margins.Max());
                int yTrue = yBin[i];

                if (yHat != yTrue)
                {
                    // 真类 +ηx
                    for (int j = 0; j < _dim; j++) _W[yTrue][j] += _lr * xb[j];
                    // 错类 -ηx
                    for (int j = 0; j < _dim; j++) _W[yHat][j] -= _lr * xb[j];
                }
            }
        }
    }
    // 预测类别概率分布
    public double[] PredictProba(double[][] X)
    {
        int n = X.Length;
        var P = new double[n * _classes];
        var outArr = new double[n * _classes];

        var res = new double[n * _classes]; // 仅为局部分配占位
        var probs = new double[_classes];
        var result = new double[n * _classes];

        var output = new double[n * _classes]; // 无用，但保留行文清晰

        var proba = new double[n * _classes];

        // 实际实现
        var R = new double[n * _classes]; // ignore
        var O = new double[n * _classes]; // ignore

        var ret = new double[n * _classes]; // ignore

        var all = new double[n * _classes]; // ignore

        var outProba = new double[n * _classes]; // ignore

        // 正确实现：返回 n x K
        var outMatrix = new double[n * _classes]; // ignore

        var out2D = new double[n * _classes]; // ignore

        var outList = new List<double[]>(); // ignore

        // 别被上面的注释干扰：下面是正确简洁实现
        var proba2D = new double[n][];
        for (int i = 0; i < n; i++)
        {
            var xb = AddBias(X[i]);
            var margins = new double[_classes];
            for (int k = 0; k < _classes; k++)
            {
                double s = 0;
                for (int j = 0; j < _dim; j++) s += _W[k][j] * xb[j];
                margins[k] = s;
            }
            proba2D[i] = Softmax(margins);
        }
        // 返回一个真正的 n x K 矩阵
        return proba2D.SelectMany(row => row).ToArray(); // 扁平化（也可以返回 double[][]）
    }
    // 预测连续值
    public double[] PredictContinuous(double[][] X)
    {
        int n = X.Length;
        var yhat = new double[n];

        for (int i = 0; i < n; i++)
        {
            var xb = AddBias(X[i]);
            var margins = new double[_classes];
            for (int k = 0; k < _classes; k++)
            {
                double s = 0;
                for (int j = 0; j < _dim; j++) s += _W[k][j] * xb[j];
                margins[k] = s;
            }
            var p = Softmax(margins);
            double e = 0;
            for (int k = 0; k < _classes; k++)
                e += p[k] * _centers[k];
            yhat[i] = e;
        }
        return yhat;
    }
}
// 评估指标
public static class Metrics
{
    public static double MAE(double[] y, double[] yhat)
        => y.Zip(yhat, (a, b) => Math.Abs(a - b)).Average();

    public static double RMSE(double[] y, double[] yhat)
    {
        double mse = y.Zip(yhat, (a, b) => (a - b) * (a - b)).Average();
        return Math.Sqrt(mse);
    }
}
// 训练与评估
public class Trainer
{
    public static void TrainAndEval(string csvPath,
                                   string targetName = "fuel_lph",
                                   int K = 12,
                                   string binStrategy = "quantile",
                                   double lr = 0.05,
                                   int epochs = 20,
                                   double tau = 0.4,
                                   bool stochasticPred = true,
                                   double validRatio = 0.2,
                                   int seed = 2025)
    {
        var data = CsvLoader.Load(csvPath);
        Console.WriteLine($"读取样本：{data.Count} 条。");

        // 构建特征
        var (Xraw, y, cols) = FeatureBuilder.Build(data);

        // 切分训练/验证
        var rnd = new Random(seed);
        int n = Xraw.Length;
        var idx = Enumerable.Range(0, n).OrderBy(_ => rnd.Next()).ToArray();
        int cut = (int)Math.Round(n * (1 - validRatio));
        var trIdx = idx.Take(cut).ToArray();
        var vaIdx = idx.Skip(cut).ToArray();

        double[][] Xtr = trIdx.Select(i => Xraw[i]).ToArray();
        double[] ytr = trIdx.Select(i => y[i]).ToArray();
        double[][] Xva = vaIdx.Select(i => Xraw[i]).ToArray();
        double[] yva = vaIdx.Select(i => y[i]).ToArray();

        // 标准化（只用训练集拟合）
        var scaler = new StandardScaler();
        scaler.Fit(Xtr);
        Xtr = scaler.Transform(Xtr);
        Xva = scaler.Transform(Xva);

        // 分桶
        var binner = Binner.Fit(ytr, K, binStrategy);
        var ytrBin = binner.Encode(ytr);
        var yvaBin = binner.Encode(yva);

        Console.WriteLine($"桶数 K = {binner.Centers.Length}；策略 = {binStrategy}");
        Console.WriteLine($"桶中心（前几个）：{string.Join(", ", binner.Centers.Take(6).Select(v => v.ToString("F2")))} ...");

        // 模型
        var model = new ProbabilisticPerceptron(
            nFeatures: Xtr[0].Length,
            nClasses: binner.Centers.Length,
            centers: binner.Centers,
            lr: lr, tau: tau,
            stochasticPred: stochasticPred, seed: seed
        );

        // 训练
        Console.WriteLine("开始训练...");
        model.Fit(Xtr, ytrBin, epochs);
        Console.WriteLine("训练完成。");

        // 评估
        var yhatTr = model.PredictContinuous(Xtr);
        var yhatVa = model.PredictContinuous(Xva);

        double maeTr = Metrics.MAE(ytr, yhatTr);
        double rmseTr = Metrics.RMSE(ytr, yhatTr);
        double maeVa = Metrics.MAE(yva, yhatVa);
        double rmseVa = Metrics.RMSE(yva, yhatVa);

        Console.WriteLine($"[Train] MAE={maeTr:F3}  RMSE={rmseTr:F3}");
        Console.WriteLine($"[Valid] MAE={maeVa:F3}  RMSE={rmseVa:F3}");

        // —— 单条预测演示 ——（用验证集第一条）
        if (Xva.Length > 0)
        {
            var one = Xva[0];
            double pred = model.PredictContinuous(new[] { one })[0];
            Console.WriteLine($"示例：验证集第 1 条真实={yva[0]:F2}，预测≈{pred:F2}");
        }

        // 你可以在此处把 scaler 的均值/方差、列名、binner 边界等保存成 JSON，便于部署。
    }
}
// 主程序入口
class Program
{
    static void Main(string[] args)
    {
        string csv = args.Length > 0 ? args[0] : "ship_fuel.csv";

        Trainer.TrainAndEval(
            csvPath: csv,
            K: 12,                 // 桶数（8–20 之间调；越大越细，数据需求越多）
            binStrategy: "quantile",// 'quantile' 或 'uniform'
            lr: 0.05,
            epochs: 25,
            tau: 0.4,              // 温度，越小越“硬”
            stochasticPred: true,  // 训练时按概率随机决策，模拟“概率性”
            validRatio: 0.2,
            seed: 2025
        );
    }
}
