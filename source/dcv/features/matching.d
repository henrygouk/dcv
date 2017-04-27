/**
Module provides a implementations of feature matching algorithms.

Copyright: Copyright Henry Gouk 2017.

Authors: Henry Gouk

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/
module dcv.features.matching;

import dcv.features.descriptor;

interface Matcher
{
    Match[] findMatches(in Descriptor[] train, in Descriptor[] query);
}

struct Match
{
    size_t trainIndex;
    size_t queryIndex;
    float score;
}

enum MatchMetric
{
    euclidean,
    manhattan,
    hamming
}

class BFMatcher : Matcher
{
    public
    {
        this(MatchMetric metric, bool crossCheck = false)
        {
            mMetric = metric;
            mCrossCheck = crossCheck;
        }

        Match[] findMatches(in Descriptor[] train, in Descriptor[] query)
        {
            if(mCrossCheck)
            {
                auto forward = findMatchesOneWay(train, query);
                auto reverse = findMatchesOneWay(query, train);

                Match[] results;

                for(size_t i = 0; i < query.length; i++)
                {
                    auto j = forward[i].trainIndex;
                    
                    if(reverse[j].trainIndex == i)
                    {
                        results ~= forward[i];
                    }
                }

                return results;
            }
            else
            {
                return findMatchesOneWay(train, query);
            }
        }
    }

    private
    {
        MatchMetric mMetric;
        bool mCrossCheck;

        Match[] findMatchesOneWay(in Descriptor[] train, in Descriptor[] query)
        {
            import std.math : exp;
            import std.numeric : euclideanDistance;
            
            Match[] result = new Match[query.length];

            foreach(i, q; query)
            {
                float bestScore = -1.0f;
                size_t bestIndex;

                foreach(j, t; train)
                {
                    float score = 0.0f;

                    switch(mMetric)
                    {
                        case MatchMetric.hamming:
                            
                            assert(t.bytes.length == q.bytes.length);

                            for(size_t k = 0; k < t.bytes.length; k++)
                            {
                                ubyte m = t.bytes[k];
                                ubyte n = q.bytes[k];

                                for(size_t l = 0; l < 8; l++)
                                {
                                    score += ((m & 1) != (n & 1)) ? 1.0f : 0.0f;
                                    m >>= 1;
                                    n >>= 1;
                                }
                            }

                            score = exp(-score / t.bytes.length);

                            break;

                        case MatchMetric.euclidean:
                            score = exp(-euclideanDistance(t.floats, q.floats));
                            break;

                        default:
                            import std.conv;
                            throw new Exception("MatchMetric " ~ mMetric.to!string ~ " has not been implemented.");
                    }

                    if(score > bestScore)
                    {
                        bestScore = score;
                        bestIndex = j;
                    }
                }

                result[i] = Match(bestIndex, i, bestScore);
            }

            return result;
        }
    }
}