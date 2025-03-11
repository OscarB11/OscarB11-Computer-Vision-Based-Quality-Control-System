using PCB_Vision.Helpers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace PCB_Vision.Models
{
    public class Users : INotifyPropertyChanged
    {
        private int id;
        private string companyName;
        private string companyIndustry;
        private string fullName;
        private string email;
        private string password;
        private bool isAdmin;
        private bool isSelected;

        private byte[] profilePicture;

        public byte[] ProfilePicture
        {
            get => profilePicture;
            set => SetProperty(ref profilePicture, value);
        }

        private BitmapImage picture;

        public BitmapImage Picture
        {
            get => picture;
            set => SetProperty(ref picture, value);
        }



        public int Id
        {
            get => id;
            set => SetProperty(ref id, value);
        }

        public string CompanyName
        {
            get => companyName;
            set => SetProperty(ref companyName, value);
        }

        public string CompanyIndustry
        {
            get => companyIndustry;
            set => SetProperty(ref companyIndustry, value);
        }

        public string FullName
        {
            get => fullName;
            set => SetProperty(ref fullName, value);
        }

        public string Email
        {
            get => email;
            set => SetProperty(ref email, value);
        }

        public string Password
        {
            get => password;
            set => SetProperty(ref password, value);
        }

        public bool IsAdmin
        {
            get => isAdmin;
            set => SetProperty(ref isAdmin, value);
        }

        public bool IsSelected
        {
            get => isSelected;
            set => SetProperty(ref isSelected, value);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetProperty<T>(ref T storage, T value, [CallerMemberName] string propertyName = null)
        {
            if (Equals(storage, value)) return false;

            storage = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }
}
