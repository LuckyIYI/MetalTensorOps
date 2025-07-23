import Foundation

public struct StaticArray16<T> {
    public typealias Values = (
        T, T, T, T, T, T, T, T,
        T, T, T, T, T, T, T, T
    )

    private var values: Values

    public init(values: Values) {
        self.values = values
    }

    public init(repeating value: T) {
        values = (
            value, value, value, value, value, value, value, value,
            value, value, value, value, value, value, value, value
        )
    }

    init(array: [T]) {
        guard array.count == 16 else { fatalError("array size mismatch") }
        values = (
            array[0], array[1], array[2], array[3],
            array[4], array[5], array[6], array[7],
            array[8], array[9], array[10], array[11],
            array[12], array[13], array[14], array[15]
        )
    }

    public subscript(_ index: Int) -> T {
        get {
            switch index {
            case 0: return values.0
            case 1: return values.1
            case 2: return values.2
            case 3: return values.3
            case 4: return values.4
            case 5: return values.5
            case 6: return values.6
            case 7: return values.7
            case 8: return values.8
            case 9: return values.9
            case 10: return values.10
            case 11: return values.11
            case 12: return values.12
            case 13: return values.13
            case 14: return values.14
            case 15: return values.15
            default: fatalError("index out of range")
            }
        }
        set {
            switch index {
            case 0: values.0 = newValue
            case 1: values.1 = newValue
            case 2: values.2 = newValue
            case 3: values.3 = newValue
            case 4: values.4 = newValue
            case 5: values.5 = newValue
            case 6: values.6 = newValue
            case 7: values.7 = newValue
            case 8: values.8 = newValue
            case 9: values.9 = newValue
            case 10: values.10 = newValue
            case 11: values.11 = newValue
            case 12: values.12 = newValue
            case 13: values.13 = newValue
            case 14: values.14 = newValue
            case 15: values.15 = newValue
            default: fatalError("index out of range")
            }
        }
    }

    @inlinable
    public func toArray() -> [T] {
        [
            self[0], self[1], self[2], self[3], self[4], self[5], self[6], self[7],
            self[8], self[9], self[10], self[11], self[12], self[13], self[14], self[15]
        ]
    }

    public func toTuple() -> Values {
        values
    }

    @inlinable
    public func map<M>(_ transform: (T) throws -> M) rethrows -> StaticArray16<M> {
        try StaticArray16<M>(values: (
            transform(self[0]), transform(self[1]), transform(self[2]), transform(self[3]),
            transform(self[4]), transform(self[5]), transform(self[6]), transform(self[7]),
            transform(self[8]), transform(self[9]), transform(self[10]), transform(self[11]),
            transform(self[12]), transform(self[13]), transform(self[14]), transform(self[15])
        ))
    }
}

public func zip<L, R>(_ lhs: StaticArray16<L>, _ rhs: StaticArray16<R>) -> StaticArray16<(L, R)> {
    StaticArray16(values: (
        (lhs[0], rhs[0]), (lhs[1], rhs[1]), (lhs[2], rhs[2]), (lhs[3], rhs[3]),
        (lhs[4], rhs[4]), (lhs[5], rhs[5]), (lhs[6], rhs[6]), (lhs[7], rhs[7]),
        (lhs[8], rhs[8]), (lhs[9], rhs[9]), (lhs[10], rhs[10]), (lhs[11], rhs[11]),
        (lhs[12], rhs[12]), (lhs[13], rhs[13]), (lhs[14], rhs[14]), (lhs[15], rhs[15])
    ))
}

extension StaticArray16: Collection {
    public typealias Index = Int
    public var startIndex: Index { 0 }
    public var endIndex: Index { 16 }
    @inlinable
    public func index(after i: Index) -> Index { i + 1 }
    public var count: Int { 16 }
}

extension StaticArray16: Sequence {
    public struct Iterator: IteratorProtocol {
        private var index = 0
        private let array: StaticArray16<T>

        init(_ array: StaticArray16<T>) {
            self.array = array
        }

        public mutating func next() -> T? {
            guard index < array.count else { return nil }
            let value = array[index]
            index += 1
            return value
        }
    }

    public func makeIterator() -> Iterator {
        Iterator(self)
    }
}

extension StaticArray16: Equatable where T: Equatable {
    @inlinable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        zip(lhs, rhs).allSatisfy { $0.0 == $0.1 }
    }
}

extension StaticArray16: Codable where T: Codable {
    public init(from decoder: Decoder) throws {
        try self.init(array: [T](from: decoder))
    }

    public func encode(to encoder: Encoder) throws {
        try toArray().encode(to: encoder)
    }
}
