using System;
using System.IO;
using System.Linq;
using System.Data.SqlClient;

using Microsoft.ML;
using Microsoft.ML.Data;

using Microsoft.Extensions.Configuration;

namespace DiabetesPredictionApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var loader = context.Data.CreateDatabaseLoader<Patient>();

            var connectionString = GetDbConnection();
            var sqlCommand = "Select CAST(Id as REAL) as Id, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, DiabetesValue From Patient";
            var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);

            Console.WriteLine("Loading data from database...");
            var data = loader.Load(dbSource);
            var set = context.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainingData = set.TrainSet;
            var testData = set.TestSet;

            var trainingPatients = context.Data.CreateEnumerable<Patient>(trainingData, reuseRowObject: true);
            Console.WriteLine($"Training Set: {trainingPatients.Count()} patients");

            Console.WriteLine("Preparing training operations...");
            var pipeline = context.Transforms
                .CopyColumns(outputColumnName: "Label", inputColumnName: "DiabetesValue")
                .Append(context.Transforms.Concatenate("Features", "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"))
                .Append(context.Regression.Trainers.FastForest());

            Console.WriteLine($"Training process is starting. {DateTime.Now.ToLongTimeString()}");
            var model = pipeline.Fit(trainingData);
            Console.WriteLine($"Training process has finished. {DateTime.Now.ToLongTimeString()}");

            var testPatients = context.Data.CreateEnumerable<Patient>(testData, reuseRowObject: true);
            Console.WriteLine($"Test Set: {testPatients.Count()} patients");

            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(testData);
            var metrics = context.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");

            var predictionEngine = context.Model.CreatePredictionEngine<Patient, DiabetesPrediction>(model);

            var patient = new Patient()
            {
                Age = 42,
                BloodPressure = 81,
                BMI = 30.1f,
                DiabetesPedigreeFunction = 0.987f,
                Glucose = 120,
                Insulin = 100,
                Pregnancies = 1,
                SkinThickness = 26,
                Id = 0,
                DiabetesValue = 0
            };

            var prediction = predictionEngine.Predict(patient);
            Console.WriteLine($"Predicted diabetes value: {prediction.PredictedDiabetesValue:0.####}");



            Console.ReadLine();
        }

        private static string GetDbConnection()
        {
            var builder = new ConfigurationBuilder()
                                .SetBasePath(Directory.GetCurrentDirectory())
                                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);

            return builder.Build().GetConnectionString("DbConnection");
        }
    }
}
