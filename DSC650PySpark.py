from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("LassoRegression").enableHiveSupport().getOrCreate()

print("Step 1 complete")

# Step 2: Load the data from the Hive table 'mobile' into a Spark DataFrame
mobile_df = spark.sql("SELECT Sale_ID, Price_USD, Units_Sold, Revenue_USD, Customer_Rating, Sale_Month,"
                      "Sale_Year FROM mobile")

mobile_df = mobile_df.na.drop()

print("Step 2 complete")

# Step 3: Prepare the data for MLlib by assembling features into a vector
assembler = VectorAssembler(
    inputCols=["Units_Sold", "Revenue_USD", "Customer_Rating", "Sale_Month", "Sale_Year"],
    outputCol="features",
    handleInvalid="skip"
)
assembled_df = assembler.transform(mobile_df).select("features", "Price_USD")

print("Step 3 complete")

# Step 4: Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

print("Step 4 complete")

# Step 5: Initialize and train a Lasso Regression model
lasso = LinearRegression(featuresCol="features", labelCol="Price_USD", elasticNetParam=1.0)
param_grid = ParamGridBuilder().addGrid(lasso.regParam, [0.001, 0.01, 0.1, 1.0]).build()
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="Price_USD", metricName="rmse")
cross_validator = CrossValidator(estimator=lasso, estimatorParamMaps=param_grid, evaluator=evaluator, 
                                 numFolds=5)
cv_model = cross_validator.fit(train_data)
lasso_model = cv_model.bestModel

print("Step 5 complete")

# Step 6: Evaluate the model on the test data
price = lasso_model.evaluate(test_data)

print("Step 6 complete")

# Step 7: Print the model performance metrics
print(f"RMSE: {price.rootMeanSquaredError}")
print(f"R^2: {price.r2}")

# Write metrics to HBase with happybase
# Example data (row_key, column_family:column, value) populated with the metrics
data = [
    ('Sale_ID', 'mobile:rmse', str(price.rootMeanSquaredError)),
    ('Sale_ID', 'mobile:r2',   str(price.r2)),
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

print("Step 7 complete")

# Step 8: Stop the Spark session
spark.stop()