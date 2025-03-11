using Microsoft.Win32;
using Newtonsoft.Json.Linq;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using PCB_Vision.Models;
using PCB_Vision.ViewModels;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using System.Xml.Linq;
using Window = System.Windows.Window;

namespace PCB_Vision.Views
{
    /// <summary>
    /// Interaction logic for DashboardView.xaml
    /// </summary>
    public partial class DashboardView : Window
    {
        private VideoCapture capture;
        private bool isCameraRunning = false;
        private DispatcherTimer timer;
        private readonly HttpClient httpClient;
        private double minScoreThreshold = 0.3;
        private Mat lastCapturedFrame;

        private DispatcherTimer timernew;
        private int elapsedSeconds = 0;
        public DashboardView()
        {
            InitializeComponent();
            httpClient = new HttpClient { BaseAddress = new Uri("http://localhost:5000") };
        }

        private void UpdateStatus(string message, bool isLoading)
        {
            statusTextBlock.Text = message;
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            UpdateStatus("Starting camera...", true);

            if (!isCameraRunning)
            {
                capture = new VideoCapture(0); // 0 is the index of the webcam
                capture.Open(0);

                if (capture.IsOpened())
                {
                    isCameraRunning = true;
                    timer = new DispatcherTimer();
                    timer.Interval = TimeSpan.FromSeconds(1); // Adjust for balance between performance and responsiveness
                    //timer.Interval = TimeSpan.FromMilliseconds(50); // Adjust for balance between performance and responsiveness
                    timer.Tick += Timer_Tick;
                    timer.Start();
                    //txtTime.Visibility = Visibility.Visible;


                    UpdateStatus("Camera started.", false);
                }
                else
                {
                    UpdateStatus("Failed to open the camera.", false);
                    MessageBox.Show("Failed to open the camera.");
                }
            }
        }

        private void UpdateFaultsList(Dictionary<string, int> faultCounts)
        {
            faultsList.Children.Clear();
            foreach (var fault in faultCounts)
            {
                var checkBox = new CheckBox { Content = $"{fault.Key} - {fault.Value}", IsChecked = true, Margin = new Thickness(5) };
                //checkBox.Checked += FaultCheckBox_Checked;
                //checkBox.Unchecked += FaultCheckBox_Unchecked;
                faultsList.Children.Add(checkBox);
            }
        }

        private void PassButton_Click(object sender, RoutedEventArgs e)
        {
            // Handle Pass button click event
            MessageBox.Show("Pass button clicked");
        }

        private void FailButton_Click(object sender, RoutedEventArgs e)
        {
            // Handle Fail button click event
            MessageBox.Show("Fail button clicked");
        }

        private void ReportButton_Click(object sender, RoutedEventArgs e)
        {
            // Handle Report button click event
            //MessageBox.Show("Report button clicked");
            InputWindow inputWindow = new InputWindow();
            inputWindow.ShowDialog();
        }

        private async void Timer_Tick(object sender, EventArgs e)
        {
            // Update the elapsed time
            elapsedSeconds++;

            // Calculate minutes and seconds
            int minutes = elapsedSeconds / 60;
            int seconds = elapsedSeconds % 60;

            // Format the time as MM:SS
            string formattedTime = $"{minutes:D2}:{seconds:D2}";

            // Display the formatted time in the TextBlock
            txtTime.Text = $"{formattedTime}";

            if (isCameraRunning && capture.IsOpened())
            {
                using (var frame = new Mat())
                {
                    capture.Read(frame);
                    if (!frame.Empty())
                    {
                        lastCapturedFrame = frame.Resize(new OpenCvSharp.Size(640, 480)); // Resize to smaller size for faster processing
                        try
                        {
                            await ProcessFrameAsync(lastCapturedFrame);
                        }
                        catch (Exception ex)
                        {
                            UpdateStatus($"Error during detection: {ex.Message}", false);
                            MessageBox.Show($"Error during detection: {ex.Message}");
                        }
                    }
                }
            }
        }

        private async Task<(Mat, Dictionary<string, int>)> DetectObjectsAsync(Mat frame)
        {
            byte[] imageBytes;
            using (var ms = new MemoryStream())
            {
                // Encode the image as a byte array (e.g., as JPEG)
                Cv2.ImEncode(".jpg", frame, out imageBytes);
            }

            var content = new ByteArrayContent(imageBytes);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");

            HttpResponseMessage response;
            try
            {
                response = await httpClient.PostAsync("/detect", content);
                response.EnsureSuccessStatusCode();
            }
            catch (Exception ex)
            {
                throw new Exception("Failed to get a response from the detection server.", ex);
            }

            var resultJson = await response.Content.ReadAsStringAsync();
            var resultData = JObject.Parse(resultJson);
            var faultCounts = resultData["faultCounts"].ToObject<Dictionary<string, int>>();
            var imgBase64 = resultData["image"].ToString();
            byte[] imgBytes = Convert.FromBase64String(imgBase64);

            var resultImage = Cv2.ImDecode(imgBytes, ImreadModes.Color);

            return (resultImage, faultCounts);
        }


        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            UpdateStatus("Stopping camera...", true);

            if (isCameraRunning)
            {
                isCameraRunning = false;
                timer.Stop();
                txtTime.Text = $"00:00";
                //txtTime.Visibility = Visibility.Collapsed;
                capture.Release();
                videoDisplay.Source = null;

                UpdateStatus("Camera stopped.", false);
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            if (isCameraRunning)
            {
                isCameraRunning = false;
                timer.Stop();
                capture.Release();
            }
        }

        private void ComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ComboBox comboBox = (ComboBox)sender;
            int selectedIndex = comboBox.SelectedIndex;

            if (captureButton != null)
            {
                // Use the selectedIndex as needed
                if (selectedIndex == 0)
                {
                    captureButton.Visibility = Visibility.Collapsed;
                }
                else if (selectedIndex == 1)
                {
                    captureButton.Visibility = Visibility.Visible;
                }
            }
        }

        private async void CheckBox_Click(object sender, RoutedEventArgs e)
        {
            CheckBox comboBox = sender as CheckBox;
            DashboardViewModel dashboardViewModel = this.DataContext as DashboardViewModel;
            Users user = comboBox.Tag as Users;
            await dashboardViewModel.SetLoginPermision(user.IsSelected, user.Id);
        }

        private async void CaptureButton_Click(object sender, RoutedEventArgs e)
        {
            if (isCameraRunning && capture.IsOpened())
            {
                UpdateStatus("Capturing image...", true);

                using (var frame = new Mat())
                {
                    capture.Read(frame);
                    if (!frame.Empty())
                    {
                        lastCapturedFrame = frame.Resize(new OpenCvSharp.Size(640, 480)); // Resize to smaller size for faster processing
                        try
                        {
                            await ProcessFrameAsync(lastCapturedFrame);
                        }
                        catch (Exception ex)
                        {
                            UpdateStatus($"Error during detection: {ex.Message}", false);
                            MessageBox.Show($"Error during detection: {ex.Message}");
                        }
                    }
                }

                UpdateStatus("Image captured.", false);
            }
        }
        private async Task ProcessFrameAsync(Mat frame)
        {
            UpdateStatus("Processing frame...", true);

            var (resultImage, faultCounts) = await DetectObjectsAsync(frame);

            videoDisplay.Dispatcher.Invoke(() =>
            {
                videoDisplay.Source = BitmapSourceConverter.ToBitmapSource(resultImage);
            });

            UpdateFaultsList(faultCounts);
            UpdateStatus("Frame processed.", false);
        }


        private async void UploadButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image files (*.jpg, *.jpeg, *.png) | *.jpg; *.jpeg; *.png";
            if (openFileDialog.ShowDialog() == true)
            {
                UpdateStatus("Uploading image...", true);

                string filePath = openFileDialog.FileName;
                try
                {
                    byte[] imageBytes = File.ReadAllBytes(filePath);
                    var content = new ByteArrayContent(imageBytes);
                    content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");

                    HttpResponseMessage response;
                    try
                    {
                        response = await httpClient.PostAsync($"/detect?threshold={minScoreThreshold}", content);
                        response.EnsureSuccessStatusCode();
                    }
                    catch (Exception ex)
                    {
                        UpdateStatus($"Failed to get a response from the detection server: {ex.Message}", false);
                        throw new Exception("Failed to get a response from the detection server.", ex);
                    }

                    var resultJson = await response.Content.ReadAsStringAsync();
                    var resultData = JObject.Parse(resultJson);
                    var faultCounts = resultData["faultCounts"].ToObject<Dictionary<string, int>>();
                    var imgBase64 = resultData["image"].ToString();
                    byte[] imgBytes = Convert.FromBase64String(imgBase64);

                    var resultImage = Cv2.ImDecode(imgBytes, ImreadModes.Color);
                    lastCapturedFrame = resultImage;
                    videoDisplay.Source = BitmapSourceConverter.ToBitmapSource(resultImage);

                    UpdateFaultsList(faultCounts);
                }
                catch (Exception ex)
                {
                    UpdateStatus($"Error during upload: {ex.Message}", false);
                    MessageBox.Show($"Error during upload: {ex.Message}");
                }

                UpdateStatus("Image uploaded.", false);
            }


        }
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Application.Current.Shutdown();
        }
        private async void ScoreThresholdSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (scoreThresholdValue != null) // Ensure scoreThresholdValue is not null
            {
                minScoreThreshold = e.NewValue;
                scoreThresholdValue.Text = minScoreThreshold.ToString("F2");
                if (lastCapturedFrame != null)
                {
                    UpdateStatus("Updating threshold...", true);
                    try
                    {
                        await ProcessFrameAsync(lastCapturedFrame);
                    }
                    catch (Exception ex)
                    {
                        UpdateStatus($"Error during detection: {ex.Message}", false);
                        MessageBox.Show($"Error during detection: {ex.Message}");
                    }
                    UpdateStatus("Threshold updated.", false);
                }
            }
        }
    }
}