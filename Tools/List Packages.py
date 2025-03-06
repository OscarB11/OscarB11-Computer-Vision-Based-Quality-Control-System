import subprocess

def list_installed_packages():
    # Run 'pip list' command and capture its output
    result = subprocess.run(['pip', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if the command was executed successfully
    if result.returncode == 0:
        # Output the result
        print("Installed packages and their versions:")
        print(result.stdout)
    else:
        # If there was an error, output the error message
        print("Error listing installed packages:")
        print(result.stderr)

list_installed_packages()
