#if os(macOS)
#else
import Foundation
#if canImport(COpenBlas)
import COpenBlas
#else
import COpenBlas86_64
#endif
// Optional
private typealias __CLPK_integer = Int32
typealias vDSP_Stride = Int
typealias vDSP_Length = UInt

func vDSP_vaddD(
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Double>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) -> Void {
    for i in 0...__N - 1 {
        let idx = Int(i) * __IB
        let outIdx = Int(i) * __IC
        __C[outIdx] = __B[idx]
    }
    cblas_daxpy(Int32(__N), 1, __A, Int32(__IA), __C, Int32(__IB))
}

func vDSP_vsubD(
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Double>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) -> Void {
    for i in 0...__N - 1 {
        let idx = Int(i) * __IB
        let outIdx = Int(i) * __IC
        __C[outIdx] = -__B[idx]
    }
    cblas_daxpy(Int32(__N), 1, __A, Int32(__IA), __C, Int32(__IB))
}

func vDSP_vsmulD(
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Double>,
    _ __C: UnsafeMutablePointer<Double>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    for i in 0...__N - 1 {
        let idx = Int(i) * __IA
        let outIdx = Int(i) * __IC
        __C[outIdx] = __A[idx]
    }

    cblas_dscal(Int32(__N), __B[0], __C, Int32(__IC))
}

func vDSP_dotprD(
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Double>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __N: vDSP_Length
) {
    __C.pointee = cblas_ddot(Int32(__N), __A, Int32(__IA), __B, Int32(__IB))
}

func vDSP_vdivD(
    _ __B: UnsafePointer<Double>,
    _ __IB: vDSP_Stride,
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __IC: vDSP_Stride,
    _ __N: vDSP_Length
) {
    var A: [Double] = [Double]()
    for i in 0 ... __N - 1 {
        let elementA = __A[Int(i) * __IA]
        let elementB = __B[Int(i) * __IB]
        __C[Int(i) * __IC] = elementA / elementB;
    }
}

func vDSP_mtransD(
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __IC: vDSP_Stride,
    _ __M: vDSP_Length,
    _ __N: vDSP_Length
) {
    var A: [Float] = [Float]()
    var eCount = 0;
    for i in 0 ... __N - 1 {
        for j in 0 ... __M - 1 {
            A.insert(Float(__A[(Int(eCount) * __IA)]), at: 0)
            eCount += 1
        }
    }

    var C = UnsafeMutablePointer<Float>.allocate(capacity: eCount)

    cblas_somatcopy(CblasRowMajor, CblasTrans, Int32(__N), Int32(__M), 1.0, A, Int32(__M), C, Int32(__N)); 
    var uCount = 0
    for i in 0 ... __M - 1 {
        for j in 0 ... __N - 1 {
            eCount -= 1
            __C[(uCount) * __IA] = (Double(C[(Int(eCount) * __IA)]))
            uCount += 1
        }
    }
}

func vDSP_mmulD(
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __B: UnsafePointer<Double>,
    _ __IB: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __IC: vDSP_Stride,
    _ __M: vDSP_Length,
    _ __N: vDSP_Length,
    _ __P: vDSP_Length
) { 
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(__M), Int32(__N), Int32(__P), 1, __A, Int32(__P), __B, Int32(__N), 0, __C, Int32(__N))
}

func vvdiv(
    _ __C: UnsafeMutablePointer<Double>,
    _ __A: UnsafePointer<Double>,
    _ __B: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        let elementA = __A[Int(i)]
        let elementB = __B[Int(i)]
        __C[Int(i)] = elementA / elementB;
    }
}

func vvdivf(
    _ __C: UnsafeMutablePointer<Float>,
    _ __A: UnsafePointer<Float>,
    _ __B: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        let elementA = __A[Int(i)]
        let elementB = __B[Int(i)]
        __C[Int(i)] = elementA / elementB;
    }
}

func vvatanh(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = atanh(__X[Int(i)]);
    }
}

func vvatanhf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = atanh(__X[Int(i)]);
    }
}

func vvacosh(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = acosh(__X[Int(i)]);
    }
}

func vvacoshf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = acosh(__X[Int(i)]);
    }
}

func vvasinh(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = asinh(__X[Int(i)]);
    }
}

func vvasinhf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = asinh(__X[Int(i)]);
    }
}

func vvtanh(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = tanh(__X[Int(i)]);
    }
}

func vvtanhf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = tanh(__X[Int(i)]);
    }
}

func vvcosh(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = cosh(__X[Int(i)]);
    }
}

func vvcoshf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = cosh(__X[Int(i)]);
    }
}

func vvsinh(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = sinh(__X[Int(i)]);
    }
}

func vvsinhf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = sinh(__X[Int(i)]);
    }
}

func vvatan(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = atan(__X[Int(i)]);
    }
}

func vvatanf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = atan(__X[Int(i)]);
    }
}

func vvacos(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = acos(__X[Int(i)]);
    }
}

func vvacosf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = acos(__X[Int(i)]);
    }
}

func vvasin(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = asin(__X[Int(i)]);
    }
}

func vvasinf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = asin(__X[Int(i)]);
    }
}

func vvsin(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = sin(__X[Int(i)]);
    }
}

func vvsinf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = sin(__X[Int(i)]);
    }
}

func vvcos(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = cos(__X[Int(i)]);
    }
}

func vvcosf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = cos(__X[Int(i)]);
    }
}

func vvtan(
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = tan(__X[Int(i)]);
    }
}

func vvtanf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = tan(__X[Int(i)]);
    }
}

func vvsincos(
    _ __Z: UnsafeMutablePointer<Double>,
    _ __Y: UnsafeMutablePointer<Double>,
    _ __X: UnsafePointer<Double>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Z[Int(i)] = sin(__X[Int(i)]);
        __Y[Int(i)] = cos(__X[Int(i)]);
    }
}

func vvsincosf(
    _ __Z: UnsafeMutablePointer<Float>,
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Z[Int(i)] = sin(__X[Int(i)]);
        __Y[Int(i)] = cos(__X[Int(i)]);
    }
}

func vDSP_measqvD(
    _ __A: UnsafePointer<Double>,
    _ __I: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __N: vDSP_Length
) {
    var sum: Double = 0;
    for i in 0 ... __N - 1 {
        sum += (__A[Int(i) * __I] * __A[Int(i) * __I]);
    }

    __C.pointee = (sum / Double(__N))
}

func vDSP_measqv(
    _ __A: UnsafePointer<Float>,
    _ __I: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    var sum: Float = 0;
    for i in 0 ... __N - 1 {
        sum += (__A[Int(i) * __I] * __A[Int(i) * __I]);
    }

    __C.pointee = (sum / Float(__N))
}

func vDSP_rmsqvD(
    _ __A: UnsafePointer<Double>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Double>,
    _ __N: vDSP_Length
) {
    vDSP_measqvD(__A, __IA, __C, __N)
    __C.pointee = sqrt(__C.pointee)
}

func vDSP_rmsqv(
    _ __A: UnsafePointer<Float>,
    _ __IA: vDSP_Stride,
    _ __C: UnsafeMutablePointer<Float>,
    _ __N: vDSP_Length
) {
    vDSP_measqv(__A, __IA, __C, __N)
    __C.pointee = sqrt(__C.pointee)
}
#endif