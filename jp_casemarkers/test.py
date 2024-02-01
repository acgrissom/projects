import os

def remove_particles_from_file(input_file, output_file):
    # Define a list of particles to be removed
    particles_to_remove = ['は', 'が', 'を', 'で', 'に', 'へ', 'と']

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            # Read the contents of the file
            input_text = file.read()

            # Split the input text into lines
            lines = input_text.strip().split('\n')

            with open(output_file, 'w', encoding='utf-8') as jp_file:
                for line in lines:
                    if "助詞" in line and any(line.startswith(p) for p in particles_to_remove):
                        # do nothing 
                        continue
                    elif "EOS" in line:
                        jp_file.seek(0, os.SEEK_END)  # Move the cursor to the end of the file
                        jp_file.seek(jp_file.tell() - 1, os.SEEK_SET)  # Move back one position
                        jp_file.truncate()  # Truncate the file, removing the last character                        
                        jp_file.write("\n")
                    else:
                        # Write other lines to the jp_file
                        jp_file.write(line.split('\t')[0])
                        jp_file.write(" ")

    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

# Example usage
if __name__ == "__main__":
    import sys

    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python test.py <input_file> <output_file>")
        sys.exit(1)

    # Get the file paths from command line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Call the function with the provided file paths
    remove_particles_from_file(input_file, output_file)

