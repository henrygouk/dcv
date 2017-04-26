/**
Module introduces the API that defines Feature Descriptor utilities in the dcv library.

Copyright: Copyright Henry Gouk 2017.

Authors: Henry Gouk

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/
module dcv.features.descriptor;

import mir.ndslice;

import dcv.core.image : Image;
import dcv.features.utils : Feature;

public import dcv.features.descriptor.brief;

union Descriptor
{
    float[] floats;
    ubyte[] bytes;
}

/**
Feature descriptor extractor interface.

Each feature descriptor extraction algorithm implements this interface.
*/
interface DescriptorExtractor
{
    /**
    Compute descriptors for features in the image.

    Params:
        image = Image in which Features have been found.
        features = Features that must be described.
    */
    Descriptor[] compute(in Image image, in Feature[] features);
}