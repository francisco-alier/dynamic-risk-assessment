import subprocess
result = subprocess.run(['python', 'scripts/scoring.py'],capture_output=True)
    # Get the output of the function as a string
output_str = result.stdout.decode('utf-8')

    # Convert the output string to a float
f1_score = float(output_str)

print(f1_score)
