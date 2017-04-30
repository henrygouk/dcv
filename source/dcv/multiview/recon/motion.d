module dcv.multiview.recon.motion;

import matte.matrix;

import mir.ndslice;

import dcv.features;

auto computeFundamentalMatrix(in Feature[] points1, in Feature[] points2, size_t numIts = 200, float thresh = 3.0f)
{
    assert(points1.length == points2.length);
    assert(points1.length > 8);

    auto convert(T)(T t)
    {
        import std.algorithm : map;
        import std.array : array;

        return t.map!(x => cast(float[2])[cast(float)x.x, cast(float)x.y]).array();
    }

    auto p1 = convert(points1);
    auto p2 = convert(points2);

    import std.algorithm : map;
    import std.array : array;
    import std.random : randomCover;
    import std.range : take, zip;
    import std.typecons : tuple;

    Matrix!float vector(float[] data)
	{
		auto v = matte.matrix.vector(data.length);
		v.data[] = data[];

		return v;
	}

    auto matches = zip(p1, p2).map!(x => tuple!("p1", "p2")(x[0], x[1])).array();
    size_t[] bestConsensus;

    //Start the RANSAC loop
    for(size_t i = 0; i < numIts; i++)
    {
        //Get a random sample
        auto sample = matches.randomCover.take(8).array();

        //Estimate the model
        auto F = fmat8Point(sample.map!(x => x.p1).array(), sample.map!(x => x.p2).array());

        size_t[] consensus;

        //Determine inliers
        for(size_t j = 0; j < matches.length; j++)
        {
            auto m1 = matches[j].p1;
            auto m2 = matches[j].p2;

            auto vec1 = vector(m1 ~ 1.0f);
            auto vec2 = vector(m2 ~ 1.0f);

            auto Fvec1 = F * vec1;
            auto Ftvec2 = F.transpose * vec2;

            auto s2 = 1.0f / (Fvec1[0, 0] * Fvec1[0, 0] + Fvec1[1, 0] * Fvec1[1, 0]);
            auto d2 = m2[0] * Fvec1[0, 0] + m2[1] * Fvec1[1, 0] + Fvec1[2, 0];

            auto s1 = 1.0f / (Ftvec2[0, 0] * Ftvec2[0, 0] + Ftvec2[1, 0] * Ftvec2[1, 0]);
            auto d1 = m1[0] * Ftvec2[0, 0] + m1[1] * Ftvec2[1, 0] + Ftvec2[2, 0];

            auto err = max(d1 * d1 * s1, d2 * d2 * s2);

            if(err < thresh)
            {
                consensus ~= j;
            }
        }

        if(consensus.length > bestConsensus.length)
        {
            import std.stdio;
            writeln("Replacing ", bestConsensus.length, " with ", consensus.length, " at itr ", i);
            bestConsensus = consensus;
        }
    }

    auto consensusMatches = bestConsensus
                           .map!(x => matches[x])
                           .array();

    auto fmat = fmat8Point(consensusMatches.map!(x => x.p1).array(), consensusMatches.map!(x => x.p2).array()).data;
    auto inliers = bestConsensus;

    return tuple!("F", "inliers")(fmat, inliers);
}

private auto fmat8Point(in float[2][] points1, in float[2][] points2)
{
    assert(points1.length == points2.length);
    assert(points1.length >= 8);

    //Find mean and stddev of each point set
    float[2] mean1 = [0.0f, 0.0f];
    float[2] mean2 = [0.0f, 0.0f];

    for(size_t i = 0; i < points1.length; i++)
    {
        mean1[] += points1[i][];
        mean2[] += points2[i][];
    }

    float c = 1.0f / points1.length;
    mean1[] *= c;
    mean2[] *= c;

    float scale1 = 0.0f;
    float scale2 = 0.0f;

    import std.math : sqrt, pow;

    for(size_t i = 0; i < points1.length; i++)
    {
        scale1 += sqrt(pow(points1[i][0] - mean1[0], 2.0f) + pow(points1[i][1] - mean1[1], 2.0f));
        scale2 += sqrt(pow(points2[i][0] - mean2[0], 2.0f) + pow(points2[i][1] - mean2[1], 2.0f));
    }

    scale1 *= c;
    scale2 *= c;

    scale1 = sqrt(2.0f) / scale1;
    scale2 = sqrt(2.0f) / scale2;

    auto p1 = new float[2][points1.length];
    auto p2 = new float[2][points1.length];

    for(size_t i = 0; i < p1.length; i++)
    {
        p1[i] = (points1[i][] - mean1[]) * scale1;
        p2[i] = (points2[i][] - mean2[]) * scale2;
    }

    import matte.matrix;

    auto eqs = Matrix!float(points1.length, 9);

    for(size_t i = 0; i < points1.length; i++)
    {
        float x1 = p1[i][0];
        float x2 = p2[i][0];
        float y1 = p1[i][1];
        float y2 = p2[i][1];

        eqs.data[i * 9 .. (i + 1) * 9] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0f];
    }

    auto eqsteqs = eqs.transpose * eqs;

    auto e = eigen(eqsteqs);

    auto Finit = Matrix!float(3, 3);
    Finit.data[] = e.vectors[$ - 1][];
    auto fsvd = svd(Finit);

    auto u = Matrix!float(3, 3);
    auto s = Matrix!float(3, 3);
    auto vt = Matrix!float(3, 3);

    u.data[] = fsvd.u[];
    s[0, 0] = fsvd.s[0];
    s[1, 1] = fsvd.s[1];
    vt.data[] = fsvd.vt[];

    auto T1 = Matrix!float(3, 3);
    auto T2 = Matrix!float(3, 3);
    T1.data[] = [scale1, 0, -scale1 * mean1[0], 0, scale1, -scale1 * mean1[1], 0, 0, 1];
    T2.data[] = [scale2, 0, -scale2 * mean2[0], 0, scale2, -scale2 * mean2[1], 0, 0, 1];

    auto F = T2.transpose * (u * s * vt) * T1;
    F = F * (1.0f / F[2, 2]);

    return F;
}

private import matte.matrix;

private auto eigen(Matrix!float mat)
{
    import lapack;

    import std.array : array;

    float[] matarr = mat.data;

    int n = cast(int)mat.rows;

    auto valsReal = new float[n];
    auto valsImag = new float[n];
    auto vecs = new float[n * n];

    geev(RowMajor, 'N', 'V', n, matarr.ptr, n, valsReal.ptr, valsImag.ptr, vecs.ptr, n, vecs.ptr, n);

    import std.array;
    import std.algorithm : map;
    import std.complex : Complex;
    import std.range : chunks, zip, transposed;
    import std.typecons : tuple;

    auto vals = zip(valsReal, valsImag)
               .map!(x => Complex!float(x[0], x[1]))
               .array();

    auto vecst = new float[n * n];

    for(size_t r = 0; r < n; r++)
    {
        for(size_t c = 0; c < n; c++)
        {
            vecst[c * n + r] = vecs[r * n + c];
        }
    }

    return tuple!("values", "vectors")(vals, vecst.chunks(n).map!(x => x.array()).array());
}

private auto svd(Matrix!float mat)
{
    import lapack;
    import std.array : array;
    import std.typecons : tuple;

    auto a = mat.data;

    auto m = cast(int)mat.rows;
    auto n = cast(int)mat.columns;
    int info = 0; 
    auto s = new float[m];
    auto u = new float[m * m];
    auto vt = new float[m * n];
    auto superb = new float[m - 1];
    int output = gesvd(RowMajor, 'A', 'A', m, n, a.ptr, n, s.ptr, u.ptr, m, vt.ptr, n, superb.ptr );

    return tuple!("u", "s", "vt")(u, s, vt);
}