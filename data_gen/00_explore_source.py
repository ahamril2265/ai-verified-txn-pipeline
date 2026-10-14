df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/Volumes/fintxn/ops/raw_source/fraudTrain.csv"))

print(df.count())
df.printSchema()
display(df.limit(5))