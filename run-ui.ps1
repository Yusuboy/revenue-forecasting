# Enable strict mode for better error handling
Set-StrictMode -Version Latest

# Define the target directory
$targetDir = ".\src"

# Check if the target directory exists
if (Test-Path -Path $targetDir) {
    try {
        # Change to the target directory
        Set-Location -Path $targetDir

        # Run the Flask application in a new process to avoid blocking PowerShell
        Write-Host "Starting Flask application in $targetDir..."
        Start-Process -NoNewWindow -FilePath "flask" -ArgumentList "--app app run" -PassThru | Out-Null

        # Wait briefly to ensure the server starts (adjust time as needed)
        Start-Sleep -Seconds 2

        # Open Chrome and navigate to http://localhost:5000/
        Write-Host "Opening Chrome to access http://localhost:5000/..."
        Start-Process -FilePath "chrome" -ArgumentList "http://localhost:5000/"

    } catch {
        # Handle errors during the Flask command execution or browser launch
        Write-Error "An error occurred: $_"
        Exit 1
    }
} else {
    # If the target directory does not exist, show an error
    Write-Error "The directory $targetDir does not exist. Please ensure the path is correct."
    Exit 1
}

# Success message
Write-Host "Flask application terminated."