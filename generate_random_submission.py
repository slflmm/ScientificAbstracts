import random
import csv

# Load test_set
test_set = []
with open('test_input.csv', 'rb') as fp:
    reader = csv.reader(fp)
    test_set = list(reader)[1:]

# Write a random category to the csv file for each example in test_set
categories = ['math', 'cs', 'stat', 'physics']
with open('test_output_random.csv', 'wb') as fp:
    writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
    writer.writerow(['id', 'category'])  # write header
    for rid, _ in test_set:
        random_category = random.choice(categories)
        writer.writerow((rid, random_category))
