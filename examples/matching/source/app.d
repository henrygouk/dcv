import std.stdio;

import mir.ndslice;

import dcv.core;
import dcv.features;
import dcv.imgproc.color;
import dcv.imgproc.filter;
import dcv.io;
import dcv.plot;

import ggplotd.ggplotd;
import ggplotd.geom;
import ggplotd.aes;

void main()
{
	auto trainImg = imread("../data/stereo/left.png");
	auto queryImg = imread("../data/stereo/right.png");

	auto trainGray = trainImg.sliced.rgb2gray.asImage;
	auto queryGray = queryImg.sliced.rgb2gray.asImage;

	auto detector = new FASTDetector(100, FASTDetector.Type.FAST_9, FASTDetector.PERFORM_NON_MAX_SUPRESSION);
	auto descriptor = new BRIEFDescriptor(64);
	auto matcher = new BFMatcher(MatchMetric.hamming, true);

	auto trainFeatures = detector.detect(trainGray);
	auto queryFeatures = detector.detect(queryGray);

	auto trainDescs = descriptor.compute(trainImg, trainFeatures);
	auto queryDescs = descriptor.compute(queryImg, queryFeatures);

	auto matches = matcher.findMatches(trainDescs, queryDescs);

	auto gg = GGPlotD();

	import std.range : take;

	import std.algorithm : sort;
	matches.sort!((a, b) => a.score > b.score);

	foreach(m; matches)
	{
		auto startX = trainFeatures[m.trainIndex].x;
		auto startY = trainFeatures[m.trainIndex].y;
		auto stopX = queryFeatures[m.queryIndex].x + trainImg.width;
		auto stopY = queryFeatures[m.queryIndex].y;

		auto aes = Aes!(size_t[], "x", size_t[], "y", string[], "colour")
					   ([startX, stopX], [startY, stopY], ["blue", "blue"]);

		gg.put(geomLine(aes));
	}

	auto combined = slice!ubyte([trainImg.height, trainImg.width + queryImg.width, 3]);
	combined[0 .. $, 0 .. trainImg.width, 0 .. 3] = trainImg.sliced;
	combined[0 .. $, trainImg.width .. $, 0 .. 3] = queryImg.sliced;

	combined.plot(gg, "Matches");

	waitKey();
}
