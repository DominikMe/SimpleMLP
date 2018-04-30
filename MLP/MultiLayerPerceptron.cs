/* Copyright Dominik Messinger, dominik.messinger@gmail.com */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLP
{
	// n-layer MLP:
	// fully connected with max 8 input and max 8 output neurons
	class MultiLayerPerceptron
	{

		public double LearningRate { get; set; } = 0.8;
		public int MaxIterations { get; set; }
		public double MaxTotalError { get; set; }
		private bool allowPrint = true;

		public int[] NeuronsInLayer { get; }
		public double[] Weights { get; }
		private double[,] results;
		private double totalError;

		internal MultiLayerPerceptron(int[] topology)
		{
			NeuronsInLayer = topology;
			var connections = getNumberOfConnections();
			Weights = new double[connections];
			results = new double[NeuronsInLayer.Length, NeuronsInLayer.Max()];
		}

		private int getNumberOfConnections()
		{
			int connections = 0;
			for (int i = 0; i < NeuronsInLayer.Length - 1; i++)
				connections += NeuronsInLayer[i] * NeuronsInLayer[i + 1];
			return connections;
		}

		private bool Terminate(int? iteration, double? totalError)
		{
			if (MaxTotalError == 0 && MaxIterations == 0)
				return true;
			return totalError <= MaxTotalError || iteration >= MaxIterations;
		}

		internal void Train(byte[] inputs, byte[] desiredOutputs)
		{
			if (inputs.Length != desiredOutputs.Length)
				throw new ArgumentException();

			Console.WriteLine("=== Learning ... ===");
			int iteration = 0;
			do
			{
				totalError = 0;
				for (int i = 0; i < inputs.Length; i++)
				{
					Compute(inputs[i]);
					var meanSqError = BackPropagate(desiredOutputs[i]);
					totalError += meanSqError / inputs.Length;
				}
				Console.WriteLine("{0}: {1}", iteration, totalError);
			}
			while (!Terminate(++iteration, totalError));
		}

		internal byte Compute(byte input)
		{
			InitResults(input);
			for (int neuron = 0; neuron < NeuronsInLayer[0]; neuron++)
			{
				results[0, neuron] = (double)((input >> neuron) & 1);
			}
			for (int layer = 1; layer < NeuronsInLayer.Length; layer++)
			{
				for (int neuron = 0; neuron < NeuronsInLayer[layer]; neuron++)
				{
					for (int neuronPrev = 0; neuronPrev < NeuronsInLayer[layer - 1]; neuronPrev++)
					{
						double x = results[layer - 1, neuronPrev] * getWeight(layer - 1, neuronPrev, neuron);
						results[layer, neuron] += x;
					}
					results[layer, neuron] = activate(results[layer, neuron]);
				}
			}
			return byteOutputFromLastLayer();
		}

		// works only for max 8 output neurons
		private byte byteOutputFromLastLayer()
		{
			int output = 0;
			int lastLayer = NeuronsInLayer.Length - 1;
			for (int neuron = NeuronsInLayer[lastLayer] - 1; neuron >= 0; neuron--)
			{
				output <<= 1;
				output += (int)Math.Round(results[lastLayer, neuron]); // round to 0 or 1
			}
			return (byte)output;
		}

		internal double BackPropagate(byte desiredOutput)
		{
			double[,] errors = new double[NeuronsInLayer.Length, NeuronsInLayer.Max()];

			int lastLayer = NeuronsInLayer.Length - 1;
			double meanSquaredError = 0;
			for (int layer = lastLayer; layer > 0; layer--)
			{
				for (int neuron = 0; neuron < NeuronsInLayer[layer]; neuron++)
				{
					// compute errors
					double neuronOutput = results[layer, neuron];
					double phi = neuronOutput * (1 - neuronOutput);
					if (layer == lastLayer)
					{
						double delta = (double)((desiredOutput >> neuron) & 1) - neuronOutput;
						errors[layer, neuron] = phi * delta;
						meanSquaredError += Math.Pow(phi * delta, 2);
					}
					else
					{
						double errorWeightSum = 0;
						for (int neuronNext = 0; neuronNext < NeuronsInLayer[layer + 1]; neuronNext++)
							errorWeightSum += errors[layer + 1, neuronNext] * getWeight(layer, neuron, neuronNext);
						errors[layer, neuron] = phi * errorWeightSum;
					}
					// adjust weights
					for (int neuronPrev = 0; neuronPrev < NeuronsInLayer[layer - 1]; neuronPrev++)
					{
						double newWeight = getWeight(layer - 1, neuronPrev, neuron) + LearningRate * errors[layer, neuron] * results[layer - 1, neuronPrev];
						setWeight(layer - 1, neuronPrev, neuron, newWeight);
					}
				}
			}
			meanSquaredError /= (NeuronsInLayer[lastLayer] + 1);
			return meanSquaredError;
		}


		internal double getWeight(int layer, int neuron, int neuronNext)
		{
			if (layer >= NeuronsInLayer.Length - 1)
				throw new ArgumentException();

			int index = getWeightIndex(layer, neuron, neuronNext);
			return Weights[index];
		}

		private void setWeight(int layer, int neuron, int neuronNext, double value)
		{
			if (layer >= NeuronsInLayer.Length - 1)
				throw new ArgumentException();

			int index = getWeightIndex(layer, neuron, neuronNext);
			Weights[index] = value;
		}

		private int getWeightIndex(int layer, int neuron, int neuronNext)
		{
			int layerOffset = 0;
			for (int l = 0; l < layer; l++)
				layerOffset += NeuronsInLayer[l];
			int index = layerOffset + neuron * NeuronsInLayer[layer + 1] + neuronNext;
			return index;
		}

		private double activate(double p)
		{
			// sigmoid
			return 1 / (1 + Math.Pow(Math.E, -p));
		}

		private void InitResults(byte xs)
		{
			for (int layer = 0; layer < NeuronsInLayer.Length; layer++)
			{
				for (int neuron = 0; neuron < NeuronsInLayer[layer]; neuron++)
				{
					results[layer, neuron] = .0f;
				}
			}
		}

		internal void RandomizeWeights()
		{
			totalError = 0;
			Random rand = new Random();
			for (int i = 0; i < Weights.Length; i++)
			{
				Weights[i] = rand.NextDouble() * 2 - 1;
			}
		}

		private void printResults()
		{
			if (!allowPrint)
				return;
			Console.WriteLine("=== RESULTS ===");
			printMatrix(results);
		}

		private void printMatrix(double[,] matrix)
		{
			for (int layer = 0; layer < NeuronsInLayer.Length; layer++)
			{
				for (int neuron = 0; neuron < NeuronsInLayer[layer]; neuron++)
				{
					Console.Write(matrix[layer, neuron].ToString("F2"));
					if (neuron != NeuronsInLayer[layer] - 1)
						Console.Write(", ");
				}
				Console.Write("\n");
			}
		}
	}
}
