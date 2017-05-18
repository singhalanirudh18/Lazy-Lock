import os
letters={"E","G","M","N","P","Q","R","S","T","U","X"}

for letter in letters:
	for x in range(10):
		os.remove(letter + str(x) + ".jpg")

#os.remove("user_3_loc.csv")