using Microsoft.ML.Data;

namespace DiabetesPredictionApp
{
    public class DiabetesPrediction
    {
        [ColumnName("Score")]
        public float PredictedDiabetesValue;
    }
}
