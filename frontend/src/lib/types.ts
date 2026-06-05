export interface ItemRegion {
    page: number;
    ymin: number;
    xmin: number;
    ymax: number;
    xmax: number;
}

export interface ProjectInfo {
    project_name: string;
    drawing_date: string;
    drawing_scale: string;
}

export interface Dimensions {
    Lx: number;
    Ly: number;
    D: string;
}

export interface FoundationItem {
    type: string;
    dimensions: Dimensions;
    top_elevation?: number | null;
    top_elevation_alt?: number | null;
    rebar_x: string;
    rebar_y: string;
    remarks: string;
    classification: string;
    image_base64?: string;
}

export interface OvalGLItem {
    text: string;
    region: ItemRegion;
}

export interface SlabItem {
    type: string;
    region: ItemRegion;
    image_base64?: string;
}

export interface RegularFloor {
    elevation: string;
    count: number;
    regions: ItemRegion[];
    image_base64?: string;
}

export interface SlopedFloor {
    start_elevation: string;
    end_elevation: string;
    region: ItemRegion;
    image_base64?: string;
}

export interface PitHoleItem {
    type: string;
    top_elevation?: number | null;
    D?: number | null;
    readable: boolean;
    region?: ItemRegion;
    image_base64?: string;
}

export interface ExtractionResponse {
    project_info: ProjectInfo;
    foundation_list: FoundationItem[];
    slab_list: SlabItem[]; // Keeping this for the count or raw data
    slab_overview_base64?: string;
    table_image_base64?: string;
    oval_gl_list: OvalGLItem[];
    annotated_pages: string[];
    floor_regular_list: RegularFloor[];
    floor_sloped_list: SlopedFloor[];
    floor_overview_base64?: string;
    pit_list?: PitHoleItem[];
    is_partial?: boolean;
    partial_stage?: string;
    partial_message?: string;
    excel_ready?: boolean;
    images_pending?: boolean;
}

export interface StreamPatchPayload {
    is_patch: true;
    patch_source: "gemini_stream";
    project_info?: ProjectInfo;
    foundation_list?: FoundationItem[];
    slab_list?: SlabItem[];
    oval_gl_list?: OvalGLItem[];
    floor_regular_list?: RegularFloor[];
    floor_sloped_list?: SlopedFloor[];
    pit_list?: PitHoleItem[];
    partial_message?: string;
    excel_ready?: boolean;
    images_pending?: boolean;
}

export type ExtractionUpdate = ExtractionResponse | StreamPatchPayload;
