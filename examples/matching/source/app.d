import std.stdio;

import mir.ndslice;

import dcv.core;
import dcv.features;
import dcv.imgproc.color;
import dcv.imgproc.filter;
import dcv.imgproc.imgmanip;
import dcv.io;
import dcv.plot;

import ggplotd.ggplotd;
import ggplotd.geom;
import ggplotd.aes;

void main()
{
	auto trainImg = imread("../data/matching/1.png").sliced.scale([0.5, 0.5]).asImage;
	auto queryImg = imread("../data/matching/2.png").sliced.scale([0.5, 0.5]).asImage;

	auto trainGray = trainImg.sliced.rgb2gray.asImage;
	auto queryGray = queryImg.sliced.rgb2gray.asImage;

	auto detector = new FASTDetector(100, FASTDetector.Type.FAST_9, FASTDetector.PERFORM_NON_MAX_SUPRESSION);
	auto descriptor = new BRIEFDescriptor(128);
	auto matcher = new BFMatcher(MatchMetric.hamming, true);

	auto trainFeatures = detector.detect(trainGray);
	auto queryFeatures = detector.detect(queryGray);

	auto trainDescs = descriptor.compute(trainImg, trainFeatures);
	auto queryDescs = descriptor.compute(queryImg, queryFeatures);

	auto matches = matcher.findMatches(trainDescs, queryDescs);

	auto gg = GGPlotD();

	auto combined = slice!ubyte([trainImg.height + 300, trainImg.width + queryImg.width, 3]);
	combined[0 .. $ - 300, 0 .. trainImg.width, 0 .. 3] = trainImg.sliced;
	combined[300 .. $, trainImg.width .. $, 0 .. 3] = queryImg.sliced;

	import std.algorithm : map;
	import std.array : array;

	auto matches1 = matches
				   .map!(x => trainFeatures[x.trainIndex])
				   .array();

	auto matches2 = matches
				   .map!(x => queryFeatures[x.queryIndex])
				   .array();

	import dcv.multiview.recon;

	auto fres = computeFundamentalMatrix(matches1.array(), matches2.array(), 100);

	import std.range : zip;
	import std.typecons : tuple;
	import std.algorithm : max;

	foreach(m1, m2; fres.inliers.map!(x => tuple(matches1[x], matches2[x])))
	{
		auto colour = "blue";

		auto startX = m1.x;
		auto startY = m1.y;
		auto stopX = m2.x + trainImg.width;
		auto stopY = m2.y + 300;

		auto aes = Aes!(size_t[], "x", size_t[], "y", string[], "colour")
					   ([startX, stopX], [startY, stopY], [colour, colour]);

		gg.put(geomLine(aes));
	}

	combined.plot(gg, "Matches");

	waitKey();
}
