## MERMAID CODE ##
flowchart TD
 subgraph Input["Mentee Profile"]
    direction LR
        A["Profile Data"]
  end
 subgraph Process["Matching Engine"]
        E["TF-IDF"]
        F["Cosine Similarity"]
  end
 subgraph Output["Ranked Mentors"]
        G["Mentor 1"]
        H["Mentor 2"]
        I["Mentor 3"]
  end
    A --- E
    E --> F
    F --> G & H & I

     A:::input
     E:::process
     F:::process
     G:::output
     H:::output
     I:::output
    classDef input fill:#E3F2FD, stroke:#1565C0, color:#1A237E
    classDef process fill:#FFF3E0, stroke:#FF9800, color:#E65100
    classDef output fill:#E8F5E9, stroke:#388E3C, color:#1B5E20


