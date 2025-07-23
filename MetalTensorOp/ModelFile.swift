import Foundation

struct ModelFile: Codable {
    let metadata: Metadata?
    let mlp: MLP?

    private enum CodingKeys: String, CodingKey {
        case metadata
        case mlp
    }

    init(metadata: Metadata?, mlp: MLP?) {
        self.metadata = metadata
        self.mlp = mlp
    }

    init(from decoder: Decoder) throws {
        do {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let metadata = try container.decodeIfPresent(Metadata.self, forKey: .metadata)
            let mlp = try container.decodeIfPresent(MLP.self, forKey: .mlp)
            self.metadata = metadata
            self.mlp = mlp
        } catch {
            if let layers = try? decoder.singleValueContainer().decode([MLPParameterLayer].self) {
                self.metadata = nil
                self.mlp = MLP(layers: layers)
            } else {
                throw error
            }
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(metadata, forKey: .metadata)
        try container.encodeIfPresent(mlp, forKey: .mlp)
    }
}


struct Metadata: Codable {
    let mode: String
    
    struct ImageMetadata: Codable {
        let width: Int?
        let height: Int?
        let aspect_ratio: Float?
    }
    
    struct SDFMetadata: Codable {
        let resolution: Int?
    }
    
    let image: ImageMetadata?
    let sdf: SDFMetadata?
}
