import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

year = np.arange(1900, 2023, 1)
area = np.arange(10, 200, 1)
#area = np.arange(10, 140, 1)
distance = np.arange(0, 25, 0.1)

price = np.arange(0, 500000)

# Year
year_old = fuzz.trapmf(year, [1900, 1900, 1930, 1960])
year_medium = fuzz.trapmf(year, [1930, 1960, 1990, 2000])
year_new = fuzz.trapmf(year, [1990, 2000, 2023, 2023])

# Area
area_small = fuzz.trapmf(area, [10, 10, 30, 40])
area_medium = fuzz.trapmf(area, [30, 40, 50, 70])
area_big = fuzz.trapmf(area, [50, 70, 200, 200])

# Distance to city centre
dist_short = fuzz.trapmf(distance, [0, 0, 2, 5])
dist_medium = fuzz.trapmf(distance, [2, 5, 8, 10])
dist_long = fuzz.trapmf(distance, [8, 10, 25, 25])

# Price
price_cheap = fuzz.trapmf(price, [0, 0, 25000, 50000])
price_medium = fuzz.trapmf(price, [30000, 50000, 100000, 150000])
price_expensive = fuzz.trapmf(price, [100000, 150000, 1000000, 1000000])


fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 12))

# Year
ax0.plot(year, year_old, 'red', linewidth=2, label='Old')
ax0.plot(year, year_medium, 'orange', linewidth=2, label='Medium')
ax0.plot(year, year_new, 'green', linewidth=2, label='New')
ax0.set_title('Built')
ax0.legend()

# Area
ax1.plot(area, area_small, 'red', linewidth=2, label='Small')
ax1.plot(area, area_medium, 'orange', linewidth=2, label='Medium')
ax1.plot(area, area_big, 'green', linewidth=2, label='Big')
ax1.set_title('Area')
ax1.legend()

# Distance
ax2.plot(distance, dist_short, 'red', linewidth=2, label='Short')
ax2.plot(distance, dist_medium, 'orange', linewidth=2, label='Medium')
ax2.plot(distance, dist_long, 'green', linewidth=2, label='Long')
ax2.set_title('Distance to city centre')
ax2.legend()

# Price 
ax3.plot(price, price_cheap, 'red', linewidth=2, label='Cheap')
ax3.plot(price, price_medium, 'orange', linewidth=2, label='Medium')
ax3.plot(price, price_expensive, 'green', linewidth=2, label='Expensive')
ax3.set_title('Price')
ax3.legend()


for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
plt.tight_layout()


# Įvesties duomenys
input_year = 1910
input_area = 10
input_dist = 20

year_old = fuzz.interp_membership(year, year_old, input_year)
year_medium = fuzz.interp_membership(year, year_medium, input_year)
year_new = fuzz.interp_membership(year, year_new, input_year)

area_small = fuzz.interp_membership(area, area_small, input_area)
area_medium = fuzz.interp_membership(area, area_medium, input_area)
area_big = fuzz.interp_membership(area, area_big, input_area)

dist_short = fuzz.interp_membership(distance, dist_short, input_dist)
dist_medium = fuzz.interp_membership(distance, dist_medium, input_dist)
dist_long = fuzz.interp_membership(distance, dist_long, input_dist)

# Not calculations
not_short_distance = np.fmax(dist_medium, dist_long)
not_long_distance = np.fmax(dist_short, dist_medium)
not_big_area = np.fmax(area_small, area_medium)
not_small_area = np.fmax(area_medium, area_big)

# Pigaus buto taisykles

# 1. Senos statybos pastatas, nedidelio (vidutinis arba didelis) buto ploto ir netrumpu (vidutinis arba didelis) atstumu nuo centro.
# 2. Vidituniu metu statybos pastatas, mazo buto ploto ir su dideliu atstumu nuo centro.

rule_1 = np.fmin(year_old, np.fmin(not_big_area, not_short_distance))
rule_2 = np.fmin(year_medium, np.fmin(area_small, dist_long))

cheap_price = np.fmax(rule_1, rule_2)

cheap_price_act = np.fmin(cheap_price, price_cheap)
print('Price of cheap flat:')
print(cheap_price)

# Vidutinios buto kainos butu taisykles

# 1. Senos statybos, nedidelio (vidutinis arba didelis) buto ploto ir trumpu atstumu nuo centro.
# 2. Senos statybos pastatas, dideliu buto plotu ir netrumpu (vidutinis arba didelis) atstumu nuo centro.
# 3. Vidutinių metų statybos pastatas, mažu buto plotu ir neilgu (mazas arba vidutinis) atstumu nuo centro.
# 4. Vidutinių metų statybos pastatas, vidutiniu buto plotu ir netrumpu (vidutinis arba didelis) atstumu nuo centro.
# 5. Vidutinių metų statybos pastatas, dideliu buto plotu ir dideliu atstumu nuo centro.
# 6. Naujos statybos pastatas, su mažu buto plotu ir netrumpu (vidutinis arba didelis) atstumu nuo centro.
# 7. Naujos statybos pastatas, su nemažu (vidutinis arba didelis) buto plotu ir dideliu atstumu nuo centro.

rule_1 = np.fmin(year_old, np.fmin(not_big_area, dist_short))
rule_2 = np.fmin(year_old, np.fmin(area_big, not_short_distance))
rule_3 = np.fmin(year_medium, np.fmin(area_small, not_long_distance))
rule_4 = np.fmin(year_medium, np.fmin(area_medium, not_short_distance))
rule_5 = np.fmin(year_medium, np.fmin(area_big, dist_long))
rule_6 = np.fmin(year_new, np.fmin(area_small, not_short_distance))
rule_7 = np.fmin(year_new, np.fmin(not_small_area, dist_long))

medium_price = np.fmax(rule_1, np.fmax(rule_2, np.fmax(rule_3, np.fmax(rule_4, np.fmax(rule_5, np.fmax(rule_6, rule_7))))))

medium_price_act = np.fmin(medium_price, price_medium)
print('Price of medium flat:')
print(medium_price)

# Dideles kainos buto taisykles

# 1. Senos statybos pastatas, su dideliu buto plotu ir mažu atstumu nuo centro.
# 2. Vidutinių metų statybos pastatas, su vidutiniu buto plotu ir trumpu atstumu nuo centro.
# 3. Vidutinių metų statybos pastatas, su dideliu buto plotu ir neilgu (mažas arba vidutinis) atstumu nuo centro.
# 4. Naujos statybos pastatas, su mažu buto plotu ir trumpu atstumu nuo centro.
# 5.  Naujos statybos pastatas, su nemažu (vidutinis arba didelis) buto plotu ir neilgu (mažas arba vidutinis) atstumu nuo centro.

rule_1 = np.fmin(year_old, np.fmin(area_big, dist_short))
rule_2 = np.fmin(year_medium, np.fmin(area_medium, dist_short))
rule_3 = np.fmin(year_medium, np.fmin(area_big, not_long_distance))
rule_4 = np.fmin(year_new, np.fmin(area_small, dist_short))
rule_5 = np.fmin(year_new, np.fmin(not_small_area, not_long_distance))

expensive_price = np.fmax(rule_1, np.fmax(rule_2, np.fmax(rule_3, np.fmax(rule_4, rule_5))))

expensive_price_act = np.fmin(expensive_price, price_expensive)
print('Price of expensive flat:')
print(expensive_price)

price0 = np.zeros_like(price)

fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(price, price0, cheap_price_act, alpha=0.7)
ax0.plot(price, price_cheap, 'red', linewidth=1, linestyle='--', )

ax0.fill_between(price, price0, medium_price_act, alpha=0.7)
ax0.plot(price, price_medium, 'orange', linewidth=1, linestyle='--')

ax0.fill_between(price, price0, expensive_price_act, alpha=0.7)
ax0.plot(price, price_expensive, 'green', linewidth=1, linestyle='--')
plt.tight_layout()


# Aggregate
aggregated = np.fmax(cheap_price_act, np.fmax(medium_price_act, expensive_price_act))

# ---------- Defuzzify --------------

price_centroid = fuzz.defuzz(price, aggregated, 'centroid')

# mean of maximum
price_mom = fuzz.defuzz(price, aggregated, 'mom') 

# max of maximum
price_lom = fuzz.defuzz(price, aggregated, 'lom')

price_activation = fuzz.interp_membership(price, aggregated, price_centroid)

fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(price, price_cheap, 'red', linewidth=1, linestyle='--')
ax0.plot(price, price_medium, 'orange', linewidth=1, linestyle='--')
ax0.plot(price, price_expensive, 'green', linewidth=1, linestyle='--')
ax0.fill_between(price, price0, aggregated, alpha=0.7)
ax0.plot([price_centroid, price_centroid], [0, price_activation], linewidth=1.5, alpha=0.9)
ax0.plot([price_mom, price_mom], [0, 0.4], linewidth=1.5, alpha=0.9)
ax0.plot([price_lom, price_lom], [0, 0.4], 'red', linewidth=1.5, alpha=0.9)
ax0.set_title('Agreguoti rezultatai')

plt.tight_layout()

print("Defuzzified results:")
print("Defuzzification: Centroid")
print(np.round(price_centroid, 2))
print("")
print("Defuzzification: MOM - Mean of maximum")
print(np.round(price_mom, 2))
print("")
print("Defuzzification: LOM - Max of maximum")
print(np.round(price_lom, 2))

plt.show()