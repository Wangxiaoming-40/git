using System;
using System.Threading.Tasks;

namespace SyncAsyncDemo
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== 同步调用示例 ===");
            RunSync();   // 同步执行
            Console.WriteLine();

            Console.WriteLine("=== 异步调用示例 ===");
            await RunAsync(); // 异步执行
            Console.WriteLine("\n程序结束。");
        }

        // 模拟一个耗时任务
        static void Download(string name)
        {
            Console.WriteLine($"开始下载 {name}...");
            Task.Delay(2000).Wait(); // 模拟耗时2秒
            Console.WriteLine($"{name} 下载完成！");
        }

        // 模拟一个耗时任务（异步版）
        static async Task DownloadAsync(string name)
        {
            Console.WriteLine($"开始下载 {name}...");
            await Task.Delay(2000); // 异步等待2秒
            Console.WriteLine($"{name} 下载完成！");
        }

        // 同步执行：一个任务一个任务地等
        static void RunSync()
        {
            var start = DateTime.Now;
            Download("文件A");
            Download("文件B");
            Download("文件C");
            Console.WriteLine($"同步执行总耗时: {(DateTime.Now - start).TotalSeconds:F3} 秒");
        }

        // 异步执行：多个任务同时进行
        static async Task RunAsync()
        {
            var start = DateTime.Now;

            // 同时开始多个下载
            Task taskA = DownloadAsync("文件A");
            Task taskB = DownloadAsync("文件B");
            Task taskC = DownloadAsync("文件C");

            // 等待所有下载完成
            await Task.WhenAll( taskA, taskB, taskC);

            Console.WriteLine($"异步执行总耗时: {(DateTime.Now - start).TotalSeconds:F3} 秒");
        }
    }
}
