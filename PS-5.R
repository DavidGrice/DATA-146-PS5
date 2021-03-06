South_With_Category_Ints <- read.csv("South_With_Category_Ints.csv")

data1 <-subset(South_With_Category_Ints, category == 'A_Lot' | category == 'Not_At_All')
int_data = data1[1:25]
cats = factor(unlist(data1[26]))

adonis(int_data~cats, method='jaccard', permutations = 200)

data2 <-subset(South_With_Category_Ints, category == 'Some' | category == 'Not_At_All')
int_data2 = data2[1:25] #do not change
cats2 = factor(unlist(data2[26])) #do not change

adonis(int_data2~cats2, method='jaccard', permutations = 200)

data3 <-subset(South_With_Category_Ints, category == 'A_Lot' | category == 'Some')
int_data3 = data3[1:25] #do not change
cats3 = factor(unlist(data3[26])) #do not change

adonis(int_data3~cats3, method='jaccard', permutations = 200)

length(South_With_Category_Ints$Arkansas)
