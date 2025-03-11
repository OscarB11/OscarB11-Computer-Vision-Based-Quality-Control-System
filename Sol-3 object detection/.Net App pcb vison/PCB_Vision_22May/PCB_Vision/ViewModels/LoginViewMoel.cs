using PCB_Vision.Helpers;
using PCB_Vision.Models;
using PCB_Vision.Views;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace PCB_Vision.ViewModels
{
	public class LoginViewMoel : BaseHandler
	{
		private Visibility logInVisibility;
		public Visibility LogInVisibility
		{
			get { return logInVisibility; }
			set { logInVisibility = value; OnPropertyChanged("LogInVisibility"); }
		}

		private Visibility signInVisibility;
		public Visibility SignInVisibility
		{
			get { return signInVisibility; }
			set { signInVisibility = value; OnPropertyChanged("SignInVisibility"); }
		}

        private bool isAdmin;

        public bool IsAdmin
        {
            get { return isAdmin; }
            set { isAdmin = value; OnPropertyChanged("IsAdmin"); }
        }

        private bool isOperator;
        public bool IsOperator
        {
            get { return isOperator; }
            set { isOperator = value; OnPropertyChanged("IsOperator"); }
        }


        private Users registerUserObj;

		public Users RegisterUserObj
        {
			get { return registerUserObj; }
			set { registerUserObj = value; OnPropertyChanged("RegisterUserObj"); }
		}

        private Users logUserObj;

        public Users LogUserObj
        {
            get { return logUserObj; }
            set { logUserObj = value; OnPropertyChanged("LogUserObj"); }
        }


        public LoginView LoginViewObj { get; set; }
        public DashboardView DashboardViewObj { get; set; }

        public RelayCommand LoginCommand { get; set; }

        public RelayCommand SignupCommand { get; set; }

        public RelayCommand RegisterCommand { get; set; }
		public RelayCommand BackToLoginCommand { get; set; }
		public LoginViewMoel(LoginView loginView) 
		{
            LoginViewObj = loginView;
            LogInVisibility = Visibility.Visible; SignInVisibility = Visibility.Collapsed;
			RegisterCommand = new RelayCommand(o => OnRegister(o));
			BackToLoginCommand = new RelayCommand(o => OnBackToLogin(o));
			LoginCommand = new RelayCommand(o => OnLogin(o));
            SignupCommand = new RelayCommand(o => OnSignup(o));
            RegisterUserObj = new Users();
            LogUserObj = new Users();
        }

        private async void OnSignup(object o)
        {
            RegisterUserObj.ProfilePicture = new byte[1024];
            RegisterUserObj.IsAdmin = true;
            using (SqlConnection connection = GetSqlConnection())
            {
                try
                {
                    connection.Open();

                    string insertQuery = @"INSERT INTO Users (CompanyName, CompanyIndustry, ProfilePicture, Fullname, Email, UserPassword, IsAdmin, IsSelect)
                                           VALUES (@CompanyName, @CompanyIndustry, @ProfilePicture, @Fullname, @Email, @UserPassword, @IsAdmin, @IsSelect)";

                    using (SqlCommand command = new SqlCommand(insertQuery, connection))
                    {
                        // Adding parameters to avoid SQL injection
                        command.Parameters.AddWithValue("@CompanyName", RegisterUserObj.CompanyName);
                        command.Parameters.AddWithValue("@CompanyIndustry", RegisterUserObj.CompanyIndustry);
                        //command.Parameters.AddWithValue("@ProfilePicture", RegisterUserObj.ProfilePicture ?? (object)DBNull.Value);

                        if (RegisterUserObj.ProfilePicture != null)
                        {
                            command.Parameters.AddWithValue("@ProfilePicture", RegisterUserObj.ProfilePicture);
                        }
                        //else
                        //{
                        //    command.Parameters.AddWithValue("@ProfilePicture", DBNull.Value);
                        //}

                        command.Parameters.AddWithValue("@Fullname", RegisterUserObj.FullName);
                        command.Parameters.AddWithValue("@Email", RegisterUserObj.Email);
                        command.Parameters.AddWithValue("@UserPassword", RegisterUserObj.Password); // Ensure this is hashed
                        command.Parameters.AddWithValue("@IsAdmin", RegisterUserObj.IsAdmin);
                        command.Parameters.AddWithValue("@IsSelect", RegisterUserObj.IsSelected);

                        int rowsAffected = await command.ExecuteNonQueryAsync();

                        if (rowsAffected > 0)
                        {
                            MessageBox.Show("Success", "Account created successfully");
                            RegisterUserObj = new Users();
                            OnBackToLogin(null);
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

        private SqlConnection GetSqlConnection()
        {
            string connectionString = @"Data Source=(local);Initial Catalog=PCB_Vision;Integrated Security=True";
            SqlConnection connection = new SqlConnection(connectionString);
            return connection;
        }

        private void OnLogin(object o)
		{
            using (SqlConnection connection = GetSqlConnection())
            {
                try
                {
                    connection.Open();

                    string selectQuery = "SELECT Id, Fullname, Email, IsAdmin,IsSelect FROM Users WHERE Email = @Email AND UserPassword = @UserPassword";

                    using (SqlCommand command = new SqlCommand(selectQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Email", LogUserObj.Email);
                        command.Parameters.AddWithValue("@UserPassword", LogUserObj.Password);

                        using (SqlDataReader reader = command.ExecuteReader())
                        {
                            if (reader.Read())
                            {
                                if ((bool)reader["IsAdmin"])
                                {
                                    IsAdmin = true;
                                    IsOperator = false;
                                    LogUserObj = new Users
                                    {
                                        Id = (int)reader["Id"],
                                        FullName = reader["Fullname"].ToString(),
                                        Email = reader["Email"].ToString(),
                                        IsAdmin = (bool)reader["IsAdmin"]
                                    };
                                    if (DashboardViewObj == null)
                                    {
                                        DashboardViewObj = new DashboardView();
                                        DashboardViewObj.DataContext = new DashboardViewModel(this);
                                    }
                                    LoginViewObj?.Hide();
                                    DashboardViewObj?.Show(); 
                                }
                                else
                                {
                                    if((bool)reader["IsSelect"])
                                    {
                                        IsAdmin = false;
                                        IsOperator = true;
                                        LogUserObj = new Users
                                        {
                                            Id = (int)reader["Id"],
                                            FullName = reader["Fullname"].ToString(),
                                            Email = reader["Email"].ToString(),
                                            IsAdmin = (bool)reader["IsAdmin"]
                                        };
                                        if (DashboardViewObj == null)
                                        {
                                            DashboardViewObj = new DashboardView();
                                            DashboardViewObj.DataContext = new DashboardViewModel(this);
                                        }
                                        LoginViewObj?.Hide();
                                        DashboardViewObj?.Show();
                                    }
                                    else
                                    {
                                        MessageBox.Show("You can't access contact your admin");
                                    }
                                }
                            }
                            else
                            {
                                MessageBox.Show("Please check Login Details");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error: " + ex.Message);
                }
            }
        }
        private void OnBackToLogin(object o)
		{
			 SignInVisibility = Visibility.Collapsed; LogInVisibility = Visibility.Visible;
		}

		private void OnRegister(object o)
		{
			LogInVisibility = Visibility.Collapsed; SignInVisibility = Visibility.Visible;
		}
	}
}
