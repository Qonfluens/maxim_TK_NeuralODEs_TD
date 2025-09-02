library(readr)
d1 = read_csv("data/MaXim__raw_datasets__set1_CLEAN.csv")
d2 = read_csv("data/MaXim__raw_datasets__set2_CLEAN.csv")
d3 = read_csv("data/MaXim__raw_datasets__set3_CLEAN.csv")
d4 = read_csv("data/MaXim__raw_datasets__set4_CLEAN.csv")


dl = list(d1 = d1, d2 = d2, d3 = d3, d4 = d4)

for(i in seq_along(dl)){
    d = dl[[i]]
    print(paste("Dataset:", names(dl)[i]))
    print(paste("datapoints:", nrow(d)))
    print(paste("time-series:", nrow(unique(d["replicate"]))))
    print(paste("n mixture:", nrow(unique(d["Mixture_abcSorted"]))))
    t = unique(d3[, c("replicate", "Mixture_abcSorted")])
    print(table(t$Mixture_abcSorted))
}



