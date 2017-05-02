/**
Module provides an implementation of the BRIEF feature descriptor.

Copyright: Copyright Henry Gouk 2017.

Authors: Henry Gouk

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/
module dcv.features.descriptor.brief;

import std.algorithm : min, max;

import mir.ndslice;

import dcv.core.image : Image;
import dcv.imgproc.filter : gaussian;
import dcv.imgproc.convolution : conv;
import dcv.features.utils : Feature;
import dcv.features.descriptor;

class BRIEFDescriptor : DescriptorExtractor
{
    public
    {
        this(size_t bytes = 32, int patchSize = 48, size_t kernelSize = 9)
        {
            mTestLocations = new int[4][bytes * 8];

            float randomGaussian(float mean, float var)
            {
                import std.random;
                import std.mathspecial;

                return normalDistributionInverse(uniform(0.0f, 1.0f)) * var + mean;
            }

            for(size_t i = 0; i < mTestLocations.length; i++)
            {
                for(size_t j = 0; j < 4; j++)
                {
                    mTestLocations[i][j] = min(max(cast(int)randomGaussian(0.0f, patchSize * patchSize / 25.0f), -patchSize / 2), patchSize / 2);
                }
            }

            mKernel = gaussian!float(kernelSize / 3, kernelSize, kernelSize);
        }

        Descriptor[] compute(in Image image, in Feature[] features)
        {
            //Make sure the image only has a single channel
            //
            
            auto data = conv(image.asType!float.sliced!float, mKernel);

            import std.algorithm : map;
            import std.array : array;

            return features.map!(x => computeSingleDescriptor(data, x)).array();
        }
    }

    private
    {
        int[4][] mTestLocations;
        Slice!(Contiguous, [2], float *) mKernel;

        Descriptor computeSingleDescriptor(S)(S data, in Feature f)
        {
            auto vals = new ubyte[mTestLocations.length / 8];

            for(size_t i = 0; i < vals.length; i++)
            {
                for(size_t j = 0; j < 8; j++)
                {
                    auto t = mTestLocations[i * 8 + j];
                    t[0] += f.y;
                    t[1] += f.x;
                    t[2] += f.y;
                    t[3] += f.x;

                    t[0] = min(max(t[0], 0), data.length!0 - 1);
                    t[1] = min(max(t[1], 0), data.length!1 - 1);
                    t[2] = min(max(t[2], 0), data.length!0 - 1);
                    t[3] = min(max(t[3], 0), data.length!1 - 1);

                    vals[i] |= (data[t[0], t[1], 0] < data[t[2], t[3], 0] ? 1 : 0) << j;
                }
            }

            Descriptor ret = {bytes: vals};

            return ret;
        }
    }
}
