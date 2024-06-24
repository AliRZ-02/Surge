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

func vvacoshf(
    _ __Y: UnsafeMutablePointer<Float>,
    _ __X: UnsafePointer<Float>,
    _ __N: UnsafePointer<Int32>
) {
    for i in 0 ... __N.pointee - 1 {
        __Y[Int(i)] = acosh(__X[Int(i)]);
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
#endif