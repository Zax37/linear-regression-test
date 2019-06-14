import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

countries = {}

class Country:
    def __init__(self):
        self.wines = []
        self.happiness = 0

class Wine:
    def __init__(self, points: int, price: float):
        self.points = points
        self.price = price

with open('winemag-data_first150k.csv', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header

    for row in reader:
        country = row[1]
        if row[5]:
            if country not in countries:
                countries[country] = Country()
            if 10 < float(row[5]) < 100:  # take only wines costing 10-100 USD
                countries[country].wines.append(Wine(int(row[4]), float(row[5])))

with open('world-happiness-report-2019.csv', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header

    for row in reader:
        if row[0] in countries and row[3] and row[4]:
            countries[row[0]].happiness = int(row[3]) / int(row[4])  # positive / negative ratio

countries = {k: v for k, v in countries.items() if v.happiness > 0 and len(v.wines) > 1000}

X = list()
y = list()
X_TEST = list()
y_EXPECTED = list()
for _, country in countries.items():
    drop_at = len(country.wines) * 0.8  # select ~80% of set as learning data and ~20% as test data
    for i, wine in enumerate(country.wines):
        if i < drop_at:
            X.append([wine.points, wine.price])
            y.append([country.happiness])
        else:
            X_TEST.append([wine.points, wine.price])
            y_EXPECTED.append([country.happiness])

predictor = LinearRegression()
predictor.fit(X, y)

predictions = predictor.predict(X=X_TEST)

plt.plot(y_EXPECTED, 'r')
plt.plot(predictions, 'g')
plt.show()
