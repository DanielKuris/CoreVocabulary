import csv


# Function to extract words from CSV and save them in the required format
def extract_words_in_format(input_csv_file, output_txt_file):
    words = []

    # Open the CSV file for reading
    with open(input_csv_file, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        # Extract only the words (ignoring the numbers)
        for row in csvreader:
            if len(row) > 1:  # Ensure there are at least two columns
                words.append(row[1])  # Append the second column (word) to the list


    # Write the words in the required format to the output text file
    with open(output_txt_file, mode='w') as txtfile:
        # Properly format words into a string
        formatted_words = "', '".join(words)
        formatted_words = "'" +formatted_words + "'"
        print(formatted_words)
        txtfile.write(formatted_words)
