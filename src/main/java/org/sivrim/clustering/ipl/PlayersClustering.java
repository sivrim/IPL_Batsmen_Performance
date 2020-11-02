package org.sivrim.clustering.ipl;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.regexp_replace;
import static org.apache.spark.sql.functions.trim;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

public class PlayersClustering {
	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		SparkSession session = SparkSession.builder().appName("players").master("local[*]").getOrCreate();

		Dataset<Row> rawData = session.read().format("csv").option("header", "true").load("data/players.csv");
		rawData.show(2);

		rawData = rawData.na().drop();
		rawData.show(2);

		// String[] columns = rawData.columns();
		// for (String column : columns) {
		// System.out.println(rawData.describe(column));
		// }

		StringIndexer textIndexers = new StringIndexer().setInputCol("Player").setOutputCol("Player_index");
		rawData = textIndexers.fit(rawData).transform(rawData);
		Dataset<Row> castedData = rawData.withColumn("HS", trim(regexp_replace(col("HS"), "\\*", "")));

		castedData = castedData.withColumn("Mat", col("Mat").cast(DataTypes.IntegerType))
				.withColumn("Inns", col("Inns").cast(DataTypes.IntegerType))
				.withColumn("NO", col("NO").cast(DataTypes.IntegerType))
				.withColumn("Runs", col("Runs").cast(DataTypes.IntegerType))
				.withColumn("HS", col("HS").cast(DataTypes.IntegerType))
				.withColumn("Ave", col("Ave").cast(DataTypes.FloatType))
				.withColumn("BF", col("BF").cast(DataTypes.IntegerType))
				.withColumn("SR", col("SR").cast(DataTypes.FloatType))
				.withColumn("100", col("100").cast(DataTypes.IntegerType))
				.withColumn("50", col("50").cast(DataTypes.IntegerType))
				.withColumn("0", col("0").cast(DataTypes.IntegerType))
				.withColumn("4s", col("4s").cast(DataTypes.IntegerType))
				.withColumn("6s", col("6s").cast(DataTypes.IntegerType));

		castedData = castedData.select("Player", "Ave", "SR");

		castedData.show(2);

		List<String> numericalColumnsAsList = Arrays.asList(castedData.columns());
		ArrayList<String> numericalColumns = new ArrayList<>(numericalColumnsAsList);
		numericalColumns.remove("Player");
		String[] numericalColumnsArray = numericalColumns.toArray(new String[numericalColumns.size()]);
		VectorAssembler assembler = new VectorAssembler().setInputCols(numericalColumnsArray)
				.setOutputCol("featuresUnscaled");
		Dataset<Row> assembled = assembler.transform(castedData);
		assembled.show(2);

		MinMaxScaler scaler = new MinMaxScaler().setInputCol("featuresUnscaled").setOutputCol("features");

		MinMaxScalerModel scalerModel = scaler.fit(assembled);
		Dataset<Row> scaledData = scalerModel.transform(assembled);
		scaledData.show(2);

		BisectingKMeans kmeans = new BisectingKMeans().setK(3).setSeed(1L);
		BisectingKMeansModel model = kmeans.fit(scaledData);

		Vector[] centers = model.clusterCenters();
		System.out.println("Cluster Centers: ");
		for (Vector center : centers) {
			System.out.println(center);
		}

		
		ClusteringEvaluator evaluator = new ClusteringEvaluator();
		double silhouette = evaluator.evaluate(model.transform(scaledData));
		System.out.println("Silhouette with squared euclidean distance = " + silhouette);

		model.transform(scaledData).select("Player", "SR", "Ave", "prediction").sort("prediction").show(50);
	}
}
