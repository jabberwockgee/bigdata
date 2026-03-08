from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("LassoRegression").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'mobile' into a Spark DataFrame
mobile_df = spark.sql("SELECT Sale_ID, Brand, Model, Country, Storage, Color, Price_USD, "
                      "Units_Sold, Revenue_USD, Customer_Rating, Payment_Method, Sale_Month,"
                      "Sale_Year FROM mobile")

# Step 2.5: Turn all the strings into numeric indices
indexer = StringIndexer(inputCol="Brand", outputCol="BrandIndex")
indexed_df = indexer.fit(mobile_df).transform(mobile_df)
encoder = OneHotEncoder(inputCols=["BrandIndex"], outputCols=["BrandVec"])
encoded_df = encoder.fit(indexed_df).transform(indexed_df)

indexer2 = StringIndexer(inputCol="Model", outputCol="ModelIndex")
indexed_df2 = indexer2.fit(encoded_df).transform(encoded_df)
encoder2 = OneHotEncoder(inputCols=["ModelIndex"], outputCols=["ModelVec"])
encoded_df2 = encoder2.fit(indexed_df2).transform(indexed_df2)

indexer3 = StringIndexer(inputCol="Country", outputCol="CountryIndex")
indexed_df3 = indexer3.fit(encoded_df2).transform(encoded_df2)
encoder3 = OneHotEncoder(inputCols=["CountryIndex"], outputCols=["CountryVec"])
encoded_df3 = encoder3.fit(indexed_df3).transform(indexed_df3)

indexer4 = StringIndexer(inputCol="Storage", outputCol="StorageIndex")
indexed_df4 = indexer4.fit(encoded_df3).transform(encoded_df3)
encoder4 = OneHotEncoder(inputCols=["StorageIndex"], outputCols=["StorageVec"])
encoded_df4 = encoder4.fit(indexed_df4).transform(indexed_df4)

indexer5 = StringIndexer(inputCol="Color", outputCol="ColorIndex")
indexed_df5 = indexer5.fit(encoded_df4).transform(encoded_df4)
encoder5 = OneHotEncoder(inputCols=["ColorIndex"], outputCols=["ColorVec"])
encoded_df5 = encoder5.fit(indexed_df5).transform(indexed_df5)

indexer6 = StringIndexer(inputCol="Payment_Method", outputCol="Payment_MethodIndex")
indexed_df6 = indexer6.fit(encoded_df5).transform(encoded_df5)
encoder6 = OneHotEncoder(inputCols=["Payment_MethodIndex"], outputCols=["Payment_MethodVec"])
encoded_df6 = encoder6.fit(indexed_df6).transform(indexed_df6)

# Dropping strings so it will finish running, hopefully

df_dropped = encoded_df6.drop("Brand", "BrandIndex","Model","ModelIndex","Country",
                              "CountryIndex","Storage","StorageIndex","Color",
                              "ColorIndex","Payment_Method","Payment_MethodIndex")

# Step 3: Prepare the data for MLlib by assembling features into a vector
assembler = VectorAssembler(
    inputCols=["Sale_ID", "BrandVec", "ModelVec", "CountryVec", "StorageVec", "ColorVec",
               "Units_Sold", "Revenue_USD", "Customer_Rating", "Payment_MethodVec", "Sale_Month",
               "Sale_Year"],
    outputCol="features"
)
assembled_df = assembler.transform(df_dropped).select("features", "Price_USD")

# Step 4: Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 5: Initialize and train a Lasso Regression model
lasso = LinearRegression(featuresCol="features", labelCol="Price_USD", elasticNetParam=1.0)
param_grid = ParamGridBuilder().addGrid(lasso.regParam, [0.001, 0.01, 0.1, 1.0]).build()
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="Price_USD", metricName="rmse")
cross_validator = CrossValidator(estimator=lasso, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
cv_model = cross_validator.fit(train_data)
lasso_model = cv_model.bestModel

# Step 6: Evaluate the model on the test data
test_results = lasso_model.evaluate(test_data)

# Step 7: Print the model performance metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase with happybase (using the provided pattern) ----
# Example data (row_key, column_family:column, value) populated with the metrics
data = [
    ('metrics1', 'mobile:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'mobile:r2',   str(test_results.r2)),
]

# Function to write data to HBase inside each partition
def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('final_project')
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

# Parallelize data and apply the function with foreachPartition
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 8: Stop the Spark session
spark.stop()