import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LSTMExample {

    public static void main(String[] args) {
        int lookBack = 10;
        int batchSize = 1;
        int epochs = 20;


        double[] data = new double[1000];
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.sin(i * 0.01);
        }


        INDArray input = Nd4j.create(data, new int[]{1, data.length});
        DataSetIterator iterator = new SinWaveDataSetIterator(input, lookBack, batchSize);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(lookBack)
                        .nOut(50)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(50)
                        .nOut(1)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));


        for (int i = 0; i < epochs; i++) {
            model.fit(iterator);
        }

        INDArray testInput = input.get(NDArrayIndex.interval(data.length - lookBack, data.length));
        INDArray predicted = model.output(testInput);
        System.out.println("Tahmin edilen değer: " + predicted);
    }
}
