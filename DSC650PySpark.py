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
mobile_df = spark.sql("SELECT Sale_ID, Price_USD, "
                      "Units_Sold, Revenue_USD, Customer_Rating, Sale_Month,"
                      "Sale_Year FROM mobile")



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