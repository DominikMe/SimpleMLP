/* Copyright Dominik Messinger, dominik.messinger@gmail.com */

using System;
using System.Collections.Generic;
using System.Text;

namespace MLP
{
	class Program
	{
		static void Main(string[] args)
		{
			var mlp = new MultiLayerPerceptron(new[] { 8, 24, 1 })
			{
				LearningRate = 0.9,
				MaxTotalError = 0.0000001,
				MaxIterations = 5000
			};

			var cond = ContainsCipher7;
			byte TargetFunc(byte b) => BoolToByte(cond(b));

			var (trainingInputs, testInputs) = GenerateByteInputs(0.85);
			var trainingOutputs = GenerateByteOutputs(TargetFunc, trainingInputs);

			mlp.RandomizeWeights();
			mlp.Train(trainingInputs, trainingOutputs);

			Test(mlp, testInputs, TargetFunc);

			PrintFirstLayerWeights(mlp);

			Console.ReadKey();
		}

		private static void PrintFirstLayerWeights(MultiLayerPerceptron mlp)
		{
			var sb = new StringBuilder();
			for (var i = 0; i < mlp.NeuronsInLayer[0]; i++)
			{
				sb.Append($"{GetWeightSum(i):F2},");
			}
			sb.Remove(sb.Length - 1, 1);
			Console.WriteLine(sb);

			double GetWeightSum(int neuron)
			{
				var sum = 0d;
				for (var j = 0; j < mlp.NeuronsInLayer[1]; j++)
					sum += mlp.getWeight(0, neuron, j) / mlp.NeuronsInLayer[1];
				return sum;
			}
		}

		delegate bool Condition(byte b);
		delegate byte TargetFunction(byte b);

		private static readonly Condition DivisibleBy3 = b => ((int)b) % 3 == 0;
		private static readonly Condition Less20OrGreater90 = b => (b < 20 || b > 90);
		private static readonly Condition Equals42 = b => (b == 42);
		private static readonly Condition Equals23Or42 = b => (b == 23 || b == 42);
		private static readonly Condition ContainsCipher7 = b => (b.ToString().Contains("7"));
		private static readonly Condition IsOdd = b => ((int)b) % 2 == 1;
		private static readonly Condition IsPrime = b => Primes.Contains((int)b);

		private static readonly List<int> Primes = new List<int>() { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241 };

		private static bool ByteToBool(byte b)
		{
			if (b > 1)
				throw new ArgumentException();
			// lsb
			return (b & 1) == 1;
		}

		private static byte BoolToByte(bool b) => b ? (byte)1 : (byte)0;

		private static void Test(MultiLayerPerceptron mlp, byte[] testInputs, TargetFunction func)
		{
			Console.WriteLine("\n======= Test =======");
			var matches = 0;
			foreach (var bx in testInputs)
			{
				var by = mlp.Compute(bx);
				var correctY = func(bx);
				if (correctY == by)
				{
					matches++;
					Console.ForegroundColor = ConsoleColor.Green;
					Console.WriteLine("{0}: {1} = {2}", bx, correctY, by);
				}
				else
				{
					Console.ForegroundColor = ConsoleColor.Red;
					Console.WriteLine("{0}: {1} != {2}", bx, correctY, by);
				}
			}
			Console.ResetColor();
			Console.WriteLine("\n{0} correct out of {1} = " + (double)matches / testInputs.Length, matches, testInputs.Length);
		}

		private static readonly Random Rand = new Random();

		private static (byte[] training, byte[] test) GenerateByteInputs(double p)
		{
			if (p < 0 || p > 1)
				throw new ArgumentException();

			var training = new List<byte>((int)(p * 256));
			var test = new List<byte>(256 - training.Capacity);
			for (var i = 0; i < 256; i++)
			{
				if (Rand.Next(100) < p * 100)
					training.Add((byte)i);
				else
					test.Add((byte)i); // BitConverter and endianness check would also work, but we have only one byte anyway
			}
			return (training.ToArray(), test.ToArray());
		}

		private static byte[] GenerateByteOutputs(TargetFunction func, byte[] inputs)
		{
			var desiredOutputs = new byte[inputs.Length];
			for (var i = 0; i < inputs.Length; i++)
				desiredOutputs[i] = func(inputs[i]);
			return desiredOutputs;
		}
	}
}
