using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text.Json;
using System.Text.Json.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace signature
{
    class Program
    {
        static VectorGaussian wPosterior;
        static InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
        static List<double> foundBlockAreas = new List<double>();
        static List<double> foundAreas = new List<double>();
        static List<bool> results = new List<bool>();

        static void SetupProbabilities()
        {
            double[] blockAreas = new double[] { 7500, 14500, 14500, 9900, 18800, 12960, 12960, 12960, 16000, 16000, 16000, 12270, 9900,  6875, 3675 };
            double[] areas = new double[] { 1291, 773, 3033, 1435, 1681, 1821, 137, 225, 3259, 0, 32, 2176, 2326,  1679, 9 };
            bool[] signature = new bool[] { true, false, true, true, true, true, false, false, true, false, false, true, true,  true, false };

            //double[] blockAreas = JsonSerializer.Deserialize<double[]>( "[7500,7500,14500,14500,9900,18800,12960,12960,12960,16000,16000,16000,12270,9900,6875,12000,12600,3675,3675,3675,3675,3675,3675]");
            //double[] areas = JsonSerializer.Deserialize<double[]>(      "[1291,2757,773,3033,1435,1681,1821,137,225,2294,35,155,1253,1715,1458,1200,1733,101,813,628,914,1,9]");
            //bool[] signature = JsonSerializer.Deserialize<bool[]>("[true,true,false,true,true,true,true,false,false,true,false,false,true,true,true,true,true,false,true,true,true,false,false]");



            Vector[] xdata = new Vector[areas.Length];
            for (int i = 0; i < xdata.Length; i++)
                xdata[i] = Vector.FromArray(areas[i], blockAreas[i], 1);
            VariableArray<Vector> x = Variable.Observed(xdata);

            // Create target y  
            VariableArray<bool> y = Variable.Observed(signature, x.Range);


            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3)));
            Range j = y.Range;
            double noise = 0.1;
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]), noise) > 0;




            



              wPosterior = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Dist over w=\n" + wPosterior);

            BinaryFormatter serializer = new BinaryFormatter();
            // write to disk  
            using (FileStream stream = new FileStream("temp.bin", FileMode.Create))
            {
                serializer.Serialize(stream, wPosterior);
            }
            // read from disk  
            using (FileStream stream = new FileStream("temp.bin", FileMode.Open))
            {
                wPosterior = (VectorGaussian)serializer.Deserialize(stream);

            }



         
           
        }


        public static void BayesPointMachine(double[] areas, double[] boxAreas, Variable<Vector> w, VariableArray<bool> y)
        { // Create x vector, augmented by 1 

            double noise = 0.1;
            Range j = y.Range; Vector[] xdata = new Vector[areas.Length];
            for (int i = 0; i < xdata.Length; i++)
                xdata[i] = Vector.FromArray(areas[i], boxAreas[i], 1);
            VariableArray<Vector> x = Variable.Observed(xdata, j); // Bayes Point Machine double noise = 0.1;  
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]), noise) > 0;
        }
        static void Main(string[] args)
        {

            SetupProbabilities();
              engine = new InferenceEngine(new ExpectationPropagation());
            string root = "";

           string name = "Seller1 Sign - Image background might cause issue.png";
           var b = RunOn(root + name, new OpenCvSharp.Rect(220, 230, 250, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(220, 270, 250, 30));
          
            Debug.Assert(b == true);  
            
            Console.WriteLine(name + "  seller   " + b);
              name = "Lien Relase 1 - Signature area contains seller1 sign contents.png";

              b = RunOn(root + name, new Rect(550, 800, 290, 50));
        
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lh1   " + b);


            b = RunOn(root + name, new Rect(50, 800, 290, 50));
            Console.WriteLine(name + "  seller   " + b);
            Debug.Assert(b == true);
            name = "Seller2 sign - Signature area contains seller1 sign contents.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(120, 820, 450 - 120, 850 - 820));
            Console.WriteLine(name + "  1 owner   " + b);
             
            Debug.Assert(b == true);
            b = RunOn(root + name, new OpenCvSharp.Rect(80, 950, 550 - 80, 990 - 950));
            Console.WriteLine(name + "  notary   " + b);
            Debug.Assert(b == true);

            name = "Lien Release 1 - Sign area 1 for IN format.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(60, 930, 360, 36));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  lr1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(480, 930, 360, 36));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr2   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(480, 670, 360, 36));
           // Cv2.WaitKey();
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr3   " + b);

            name = "Lien Release 1 - Sign area 2 for IN format.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(460, 600, 400, 40));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  lr1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(460, 760, 400, 40));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr2   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(460, 900, 400, 40));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr3   " + b);

            name = "Buyer Sign - Contains only scribling.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(480, 400, 409, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(560, 270, 330, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  seller   " + b);

            name = "Seller2 sign - Signature area contains seller1 sign contents and stamp_2.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(130, 820, 275, 25));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(75, 960, 480, 25));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);


            name = "Seller2 Sign - Signature area contains seller1 sign contents3.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(20, 880, 420, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(490, 950, 245, 15));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(490, 695, 245, 15));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);

            name = "Seller1 Sign - Image background might cause issue_2.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(230, 240, 245, 15));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(230, 280, 245, 15));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  sq   " + b);

            b = RunOn(root + name, new OpenCvSharp.Rect(620, 240, 245, 15));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  cob   " + b);

            b = RunOn(root + name, new OpenCvSharp.Rect(620, 280, 245, 15));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  cos   " + b);

            string fas = JsonSerializer.Serialize( foundBlockAreas.ToArray());
            string fa = JsonSerializer.Serialize(foundAreas.ToArray());
            string rb = JsonSerializer.Serialize(results.ToArray());

            Cv2.WaitKey();

        }
        static bool RunOn(string file, OpenCvSharp.Rect boundingBox)
        {
            int areas = 0;

            int[] quadrants = new int[4];
            using (OpenCvSharp.Mat m = new OpenCvSharp.Mat(file, OpenCvSharp.ImreadModes.Grayscale))
            {
                //blur the image a little
                using (var blurred = m.GaussianBlur(new Size(3, 3), 0))
                {
                    //make the image binary black or white and make black the background color
                    using (var g = blurred.Threshold(200, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu))
                    {
                        
                        var element = Cv2.GetStructuringElement(
                                            MorphShapes.Rect,
                                            new Size(50, 1));
                        //remove lines from dark background image by creating a mask
                        using (var mask = g.MorphologyEx(MorphTypes.Open, element, iterations: 2))
                        {

                            using (Mat newMask = new Mat())
                            {

                                //mask bits should be 0 to skip copying items
                                Cv2.BitwiseNot(mask, newMask);



                                using (Mat newImage = new Mat())
                                {
                                    //make new image and apply mask so as to not copy the lines
                                    g.CopyTo(newImage, newMask);

                                    //create the box image
                                    using (OpenCvSharp.Mat box = new OpenCvSharp.Mat(new Size(boundingBox.Width, boundingBox.Height), MatType.CV_8U))
                                    {

                                        //copy to the box
                                        newImage[boundingBox].CopyTo(box);




                                        using (Mat labels = new Mat())
                                        {
                                            using (var centroids = new Mat())
                                            {
                                                using (Mat stats = new Mat())
                                                {

                                                    //find the white blobs
                                                    //populate the quadrants blobs appear in
                                                    //create total area of white stuff

                                                    int cnt = Cv2.ConnectedComponentsWithStats(box, labels, stats, centroids, PixelConnectivity.Connectivity8);
#if usequadrants

                                                    int qh = box.Size().Height / 2;
                                                    int qw = box.Size().Width / 2;

                                                    var tl = new Rect(0, 0, qw, qh);
                                                    var vl = new Rect(0, qh, qw, qh);
                                                    var tr = new Rect(qw, 0, qw, qh);
                                                    var br = new Rect(qw, qh, qw, qh);
#endif
                                                    for (var x = 1; x < stats.Size().Height; x++)
                                                    {
                                                        #if usequadrants
                                                        var left = stats.Get<int>(x, (int)ConnectedComponentsTypes.Left);
                                                        var top = stats.Get<int>(x, (int)ConnectedComponentsTypes.Top);
                                                        var width = stats.Get<int>(x, (int)ConnectedComponentsTypes.Width);
                                                        var height = stats.Get<int>(x, (int)ConnectedComponentsTypes.Height);

                                                        var re = new Rect(left, top, width, height);
                                                        if (re.IntersectsWith(tl))
                                                        {
                                                            quadrants[0] = 1;
                                                        }
                                                        if (re.IntersectsWith(vl))
                                                        {
                                                            quadrants[1] = 1;
                                                        }
                                                        if (re.IntersectsWith(tr))
                                                        {
                                                            quadrants[2] = 1;
                                                        }
                                                        if (re.IntersectsWith(br))
                                                        {
                                                            quadrants[3] = 1;
                                                        }


#endif
                                                        areas += stats.Get<int>(x, (int)ConnectedComponentsTypes.Area);

                                                    }
                                                }
                                            }
                                        }

                                        var boxarea = box.Size().Width * box.Size().Height;



                                        double[] areasTest = new double[] { areas };
                                        double[] boxAreas = new double[] { boxarea };


                                        //use infer.net to determine if the mean is good or not
                                        VariableArray<bool> ytest = Variable.Array<bool>(new Range(areasTest.Length));
                                        BayesPointMachine(areasTest, boxAreas, Variable.Random(wPosterior), ytest);
                                        var res = (DistributionStructArray<Bernoulli, bool>)engine.Infer(ytest);
                                        var mean = res[0].GetMean();


                                        Console.WriteLine(boxarea + " " + areas + " " + mean + " "
                                            #if usequadrants
                                            + quadrants.Sum());
#else
                                            );
#endif
                                        bool probableSiganture = false;
                                        if (mean > 0.5
#if usequadrants
                                        && quadrants.Sum() > 1)
#endif
                                        )
                                        {
                                            probableSiganture = true;
                                        }


                                        foundBlockAreas.Add(boxarea);
                                        foundAreas.Add(areas);
                                        results.Add(probableSiganture);



                                        //  Cv2.ImShow("box", box);
                                        //  Cv2.WaitKey();
                                        return probableSiganture;
                                    }
                                }
                            }
                        }
                    }
                }
            }

        }
    }
}
