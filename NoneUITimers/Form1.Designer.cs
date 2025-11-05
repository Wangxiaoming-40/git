namespace TimerDemo
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;
        private System.Windows.Forms.Label labelCounter;
        private System.Windows.Forms.Button btnStart;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.CheckBox chkAsync;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.labelCounter = new System.Windows.Forms.Label();
            this.btnStart = new System.Windows.Forms.Button();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.chkAsync = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // labelCounter
            // 
            this.labelCounter.AutoSize = true;
            this.labelCounter.Font = new System.Drawing.Font("Segoe UI", 16F);
            this.labelCounter.Location = new System.Drawing.Point(30, 30);
            this.labelCounter.Name = "labelCounter";
            this.labelCounter.Size = new System.Drawing.Size(25, 30);
            this.labelCounter.TabIndex = 0;
            this.labelCounter.Text = "0";
            // 
            // btnStart
            // 
            this.btnStart.Font = new System.Drawing.Font("Segoe UI", 10F);
            this.btnStart.Location = new System.Drawing.Point(30, 80);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(120, 35);
            this.btnStart.TabIndex = 1;
            this.btnStart.Text = "Start Timer";
            this.btnStart.UseVisualStyleBackColor = true;
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // timer1
            // 
            this.timer1.Interval = 1000; // 1 秒
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // chkAsync
            // 
            this.chkAsync.AutoSize = true;
            this.chkAsync.Font = new System.Drawing.Font("Segoe UI", 9F);
            this.chkAsync.Location = new System.Drawing.Point(170, 88);
            this.chkAsync.Name = "chkAsync";
            this.chkAsync.Size = new System.Drawing.Size(93, 19);
            this.chkAsync.TabIndex = 2;
            this.chkAsync.Text = "Use Async";
            this.chkAsync.UseVisualStyleBackColor = true;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 141);
            this.Controls.Add(this.chkAsync);
            this.Controls.Add(this.btnStart);
            this.Controls.Add(this.labelCounter);
            this.Name = "Form1";
            this.Text = "Sync vs Async Timer";
            this.ResumeLayout(false);
            this.PerformLayout();
        }
    }
}
