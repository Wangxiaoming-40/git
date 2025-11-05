using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TimerDemo
{
    public partial class Form1 : Form
    {
        private int _counter = 0;
        private bool _isRunning = false;

        public Form1()
        {
            InitializeComponent();
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            _counter = 0;
            labelCounter.Text = "0";
            timer1.Start();
        }

        // 定时器 Tick 事件：根据勾选决定用同步还是异步
        private async void timer1_Tick(object sender, EventArgs e)
        {
            if (chkAsync.Checked)
            {
                await RunAsyncVersion();
            }
            else
            {
                RunSyncVersion();
            }
        }

        // 同步版本 —— 会阻塞 UI，界面会卡顿
        private void RunSyncVersion()
        {
            Thread.Sleep(2000); // 等待 2 秒（阻塞 UI 线程）
            _counter++;
            labelCounter.Text = _counter.ToString();
        }

        // 异步版本 —— 不阻塞 UI，界面流畅
        private async Task RunAsyncVersion()
        {
            if (_isRunning) return; // 防止重入
            _isRunning = true;

            await Task.Delay(2000); // 异步等待 2 秒，不阻塞 UI
            _counter++;
            labelCounter.Text = _counter.ToString();

            _isRunning = false;
        }
    }
}
