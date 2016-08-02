/* Copyright Dominik Messinger, dominik.messinger@gmail.com */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    class Program
    {
        static void Main(string[] args)
        {
            MultiLayerPerceptron mlp = new MultiLayerPerceptron();

            Condition cond = IsPrime;
            TargetFunction targetFunc = b => BoolToByte(cond(b));

            byte[] inputs;
            GenerateByteInputs(out inputs, 20, 120);
            byte[] desiredOutputs;
            GenerateByteOutputs(out desiredOutputs, targetFunc, 20, 120);

            mlp.RandomizeWeights();
            mlp.MaxTotalError = 0.00001;
            mlp.MaxIterations = 1000;

            mlp.Train(inputs, desiredOutputs);

            Test(mlp, 100, targetFunc);
            Console.ReadKey();
        }

        delegate bool Condition(byte b);
        delegate byte TargetFunction(byte b);

        private static Condition DivisibleBy3 = b => ((int) b) % 3 == 0;
        private static Condition Less20OrGreater90 = b => (b < 20 || b > 90);
        private static Condition Equals42 = b => (b == 42);
        private static Condition Equals23Or42 = b => (b == 23 || b == 42);
        private static Condition ContainsCipher2 = b => (b.ToString().Contains("2"));
        private static Condition IsOdd = b => ((int)b) % 2 == 1;
        private static Condition IsPrime = b => Primes.Contains((int) b);

        private static readonly List<int> Primes = new List<int>() {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241}; 

        private static bool ByteToBool(byte b)
        {
            if (b > 1)
                throw new ArgumentException();
            // lsb
            return (b & 1) == 1;
        }

        private static byte BoolToByte(bool b)
        {
            return b ? (byte) 1 : (byte) 0;
        }

        private static void Test(MultiLayerPerceptron mlp, int n, TargetFunction func)
        {
            Console.WriteLine("\n======= Test =======");
            Random rand = new Random();
            int matches = 0;
            for(int i = 0; i < n; i++)
            {
                int x = rand.Next(256);
                byte bx = (byte) x;
                byte by = mlp.Compute(bx);
                byte correctY = func(bx);
                if(correctY == by)
                {
                    matches++;
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("{0}: {1} = {2}", x, correctY, by);
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("{0}: {1} != {2}", x, correctY, by);
                }
            }
            Console.ResetColor();
            Console.WriteLine("\n{0} correct out of {1}", matches, n);
        }

        private static void GenerateByteInputs(out byte[] inputs, int start = 0, int end = 256)
        {
            inputs = new byte[end - start];
            for (int i = start; i < end; i++)
                inputs[i - start] = byte.Parse(i.ToString()); // BitConverter and endianness check would also work, but we have only one byte anyway
        }

        private static void GenerateByteOutputs(out byte[] desiredOutputs, TargetFunction func, int start = 0, int end = 256)
        {
            desiredOutputs = new byte[end - start];
            for (int i = start; i < end; i++)
                desiredOutputs[i - start] = func((byte) i);
        }
    }
}
