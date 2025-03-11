using Microsoft.Win32;
using PCB_Vision.Helpers;
using PCB_Vision.Models;
using PCB_Vision.Views;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using System.Net.Mail;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Xml.Linq;

namespace PCB_Vision.ViewModels
{
    public class DashboardViewModel : BaseHandler
    {

        private string _selectedImagePath;

        public string SelectedImagePath
        {
            get { return _selectedImagePath; }
            set
            {
                if (_selectedImagePath != value)
                {
                    _selectedImagePath = value;
                    OnPropertyChanged("SelectedImagePath");
                    OnPropertyChanged("SelectedImage");
                }
            }
        }

        public BitmapImage SelectedImage
        {
            get
            {
                if (!string.IsNullOrEmpty(_selectedImagePath))
                {
                    return new BitmapImage(new Uri(_selectedImagePath));
                }
                return null;
            }
        }

        private Users operatorUserObj;

        public Users OperatorUserObj
        {
            get { return operatorUserObj; }
            set { operatorUserObj = value; OnPropertyChanged("OperatorUserObj"); }
        }

        public RelayCommand LogoutCommand { get; set; }
        public RelayCommand ChooseProfileCommand { get; set; }
        public RelayCommand EmailsendCommand { get; set; }
        public RelayCommand AddOperatorCommand { get; set; }
        public LoginViewMoel LoginViewModelVM { get; set; }

        private ObservableCollection<Users> usersList;

        public ObservableCollection<Users> UsersList
        {
            get { return usersList; }
            set { usersList = value; OnPropertyChanged("UsersList"); }
        }


        //public ObservableCollection<Users> UsersList { get; set; }
        public DashboardViewModel(LoginViewMoel loginViewMoel)
        {
            LoginViewModelVM = loginViewMoel;
            UsersList = new ObservableCollection<Users>();
            LogoutCommand = new RelayCommand(o => OnLogout(o));
            ChooseProfileCommand = new RelayCommand(o => OnChooseProfile(o));
            EmailsendCommand = new RelayCommand(o => OnEmailsend(o));
            AddOperatorCommand = new RelayCommand(o => OnAddOperator());
            OperatorUserObj = new Users();
            OperatorUserObj.ProfilePicture = new byte[1024];
            LoadUsers();
        }

        private async void OnAddOperator()
        {
            //OperatorUserObj.IsSelected = true;
            using (SqlConnection connection = GetSqlConnection())
            {
                try
                {
                    connection.Open();

                    string insertQuery = @"INSERT INTO Users (ProfilePicture, Fullname, Email, UserPassword, IsAdmin, IsSelect)
                                           VALUES (@ProfilePicture, @Fullname, @Email, @UserPassword, @IsAdmin, @IsSelect)";

                    using (SqlCommand command = new SqlCommand(insertQuery, connection))
                    {
                        // Adding parameters to avoid SQL injection
                        //command.Parameters.AddWithValue("@CompanyName", OperatorUserObj.CompanyName);
                        //command.Parameters.AddWithValue("@CompanyIndustry", OperatorUserObj.CompanyIndustry);
                        //command.Parameters.AddWithValue("@ProfilePicture", RegisterUserObj.ProfilePicture ?? (object)DBNull.Value);

                        if (OperatorUserObj.ProfilePicture != null)
                        {
                            if(SelectedImage!=null)
                            {
                                OperatorUserObj.ProfilePicture = BitmapToByteConvert(SelectedImage);
                            }
                            command.Parameters.AddWithValue("@ProfilePicture", OperatorUserObj.ProfilePicture);
                        }
                        //else
                        //{
                        //    command.Parameters.AddWithValue("@ProfilePicture", DBNull.Value);
                        //}

                        command.Parameters.AddWithValue("@Fullname", OperatorUserObj.FullName);
                        command.Parameters.AddWithValue("@Email", OperatorUserObj.Email);
                        command.Parameters.AddWithValue("@UserPassword", OperatorUserObj.Password); // Ensure this is hashed
                        command.Parameters.AddWithValue("@IsAdmin", OperatorUserObj.IsAdmin);
                        command.Parameters.AddWithValue("@IsSelect", OperatorUserObj.IsSelected);

                        int rowsAffected = await command.ExecuteNonQueryAsync();

                        if (rowsAffected > 0)
                        {
                            MessageBox.Show("Success", "Account created successfully");
                            OperatorUserObj = new Users();
                            LoadUsers();
                        }
                        else
                        {
                            MessageBox.Show("Failed", "Account creation failed, please try again");
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error", $"An error occurred: {ex.Message}");
                }
            }
        }

        private byte[] BitmapToByteConvert(BitmapImage selectedImage)
        {
            byte[] data;
            JpegBitmapEncoder encoder = new JpegBitmapEncoder(); // or use PngBitmapEncoder, BmpBitmapEncoder, etc.
            encoder.Frames.Add(BitmapFrame.Create(selectedImage));

            using (MemoryStream ms = new MemoryStream())
            {
                encoder.Save(ms);
                data = ms.ToArray();
            }

            return data;
        }

        private SqlConnection GetSqlConnection()
        {
            string connectionString = @"Data Source=(local);Initial Catalog=PCB_Vision;Integrated Security=True";
            SqlConnection connection = new SqlConnection(connectionString);
            return connection;
        }

        private async void OnEmailsend(object o)
        {
            try
            {
                Users users = o as Users;
                
                await Task.Run(async () =>
                {
                    await MailSend(users.Email);
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message.ToString());
            }
        }

        private async Task MailSend(string email)
        {
            try
            {
                MailMessage mail = new MailMessage();
                SmtpClient smtpServer = new SmtpClient("smtp.gmail.com");
                smtpServer.Port = 587;
                smtpServer.Credentials = new System.Net.NetworkCredential("aplusgamer987@gmail.com", "axao mzcv yync qvqt"); smtpServer.EnableSsl = true;


                mail.From = new MailAddress("aplusgamer987@gmail.com");
                mail.To.Add(email);
                mail.Subject = "PCB Vision operator session";
                mail.Body = "You have been selected by your manager for pcb testing report to your station";

               
                smtpServer.Send(mail);
                MessageBox.Show("Mail sent successfully.");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message.ToString());
            }
        }

        public async Task SetLoginPermision(bool isSelected,int id)
        {
            using (SqlConnection connection = GetSqlConnection())
            {
                try
                {
                    connection.Open();

                    // Check if the OperatorUserObj has an ID
                    if (id != 0)
                    {
                        string updateQuery = @"UPDATE Users SET IsSelect = @IsSelect WHERE Id = @Id";

                        using (SqlCommand command = new SqlCommand(updateQuery, connection))
                        {
                            command.Parameters.AddWithValue("@IsSelect", isSelected);
                            command.Parameters.AddWithValue("@Id", id);

                            int rowsAffected = await command.ExecuteNonQueryAsync();

                            if (rowsAffected > 0)
                            {
                                //MessageBox.Show("Success", "IsSelected updated successfully");
                                // Reload users or update UI as needed
                            }
                            else
                            {
                                //MessageBox.Show("Failed", "Failed to update IsSelected");
                            }
                        }
                    }
                    else
                    {
                        MessageBox.Show("Error", "OperatorUserObj has no valid ID");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error", $"An error occurred: {ex.Message}");
                }
            }
        }

        private void OnChooseProfile(object o)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image files (*.jpg, *.jpeg, *.png) | *.jpg; *.jpeg; *.png";

            if (openFileDialog.ShowDialog() == true)
            {
                SelectedImagePath = openFileDialog.FileName;
                //ProfilePicture.Source = new BitmapImage(new Uri(selectedFileName));
            }
        }

        private void LoadUsers()
        {
            UsersList = new ObservableCollection<Users>();
            using (SqlConnection connection = GetSqlConnection())
            {
                connection.Open();
                string selectQuery = "SELECT Id, ProfilePicture, Fullname, Email, UserPassword, IsAdmin, IsSelect FROM Users";
                using (SqlCommand command = new SqlCommand(selectQuery, connection))
                {
                    using (SqlDataReader reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            bool isAdming = (bool)reader["IsAdmin"];

                            if(!isAdming)
                            {
                                Users user = new Users
                                {
                                    Id = (int)reader["Id"],
                                    // Assuming ProfilePicture is stored as byte[]
                                    ProfilePicture = (byte[])reader["ProfilePicture"],
                                    Picture = ByteArrayToBitmapImage((byte[])reader["ProfilePicture"]),
                                    FullName = reader["Fullname"].ToString(),
                                    Email = reader["Email"].ToString(),
                                    Password = reader["UserPassword"].ToString(),
                                    IsAdmin = (bool)reader["IsAdmin"],
                                    IsSelected = (bool)reader["IsSelect"]
                                };
                                UsersList.Add(user);
                            }
                        }
                    }
                }
                //UsersList.Add(new Users { FullName ="Nilesh Chaudhari",Email = "chaudharinilesh56@gmail.com",Password="nilesh@1234",IsSelected =false,IsAdmin=false});
                //UsersList.Add(new Users { FullName ="vkey Chaudhari",Email = "vkey@gmail.com",Password="vkey@1234",IsSelected =true,IsAdmin=false});
            }
        }

        public BitmapImage ByteArrayToBitmapImage(byte[] byteArray)
        {
            if (byteArray == null || byteArray.Length == 0)
                return null;

            BitmapImage bitmapImage = new BitmapImage();

            using (MemoryStream memoryStream = new MemoryStream(byteArray))
            {
                memoryStream.Position = 0; // Ensure stream is at the beginning
                bitmapImage.BeginInit();
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.StreamSource = memoryStream;
                bitmapImage.EndInit();
            }

            return bitmapImage;
        }

        private void OnLogout(object o)
        {
            LoginViewModelVM.DashboardViewObj?.Hide();
            LoginViewModelVM.LoginViewObj?.Show();
        }

    }
}
