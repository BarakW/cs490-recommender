import csv

RT_file = open('reviews.csv', encoding="utf-8")
RT_reader = csv.reader(RT_file)

with open('clean_RT.csv', 'w', newline="", encoding="utf-8") as cleaned_RT_file:
    cleaned_RT_writer = csv.writer(cleaned_RT_file)
    first_read_row = next(RT_reader)
    cleaned_RT_writer.writerow([first_read_row[0], first_read_row[1], first_read_row[4], first_read_row[2]])

    for row in RT_reader:
        try:
            numer, denom = row[4].split("/")
            numer_rating = int(100 * float(numer) / float(denom))
            new_row = [row[0], "/movie" + row[1][2:], numer_rating, row[2]]
            cleaned_RT_writer.writerow(new_row)
        except:
            pass

RT_file.close()