import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SimpleNN {
    public static void main(String[] args) {
        // Sinir ağı modelini oluştur
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(2).nOut(4)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .nIn(4).nOut(1).build())
                .build());
        model.init();

        // Modeli eğitme (kapsamlı veri seti ve eğitim döngüsü olmadan basit bir çağrı)
        // model.fit(dataSet);  // Burada örnek veri setini eğitmek için kullanılacak

        System.out.println("Sinir ağı modeli başarıyla oluşturuldu.");
    }
}
