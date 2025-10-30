from typing import List

from fetchcraft.node import Chunk

AI_CHUNKS: List[Chunk] = [
    Chunk.from_text(
        text="Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves through algorithms.",
        chunk_index=0,
        metadata={"topic": "machine_learning", "category": "fundamentals"}
    ),
    
    Chunk.from_text(
        text="Deep Learning uses neural networks with multiple layers to progressively extract higher-level features from raw input. For example, in image recognition, lower layers identify edges, while higher layers identify human-relevant concepts like digits or letters or faces.",
        chunk_index=1,
        metadata={"topic": "deep_learning", "category": "neural_networks"}
    ),
    
    Chunk.from_text(
        text="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, to bridge the gap between human communication and computer understanding.",
        chunk_index=2,
        metadata={"topic": "nlp", "category": "language"}
    ),
    
    Chunk.from_text(
        text="Computer Vision enables computers to derive meaningful information from digital images, videos, and other visual inputs. It seeks to automate tasks that the human visual system can do, such as recognizing objects, tracking movements, and understanding scenes.",
        chunk_index=3,
        metadata={"topic": "computer_vision", "category": "perception"}
    ),
    
    Chunk.from_text(
        text="Reinforcement Learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving feedback in the form of rewards or penalties.",
        chunk_index=4,
        metadata={"topic": "reinforcement_learning", "category": "learning_paradigms"}
    ),
    
    Chunk.from_text(
        text="Generative AI refers to artificial intelligence systems that can create new content, including text, images, music, audio, and videos. Models like GPT and DALL-E are examples of generative AI that have transformed content creation and creative workflows.",
        chunk_index=5,
        metadata={"topic": "generative_ai", "category": "content_generation"}
    ),
    
    Chunk.from_text(
        text="Transfer Learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. It leverages knowledge gained from solving one problem and applies it to a different but related problem, significantly reducing training time and data requirements.",
        chunk_index=6,
        metadata={"topic": "transfer_learning", "category": "optimization"}
    ),
    
    Chunk.from_text(
        text="AI Ethics addresses the moral implications and societal impact of artificial intelligence systems. Key concerns include algorithmic bias, privacy, transparency, accountability, and the potential displacement of human workers. Establishing ethical guidelines is crucial for responsible AI development.",
        chunk_index=7,
        metadata={"topic": "ai_ethics", "category": "ethics"}
    ),
    
    Chunk.from_text(
        text="Explainable AI (XAI) focuses on creating AI systems whose actions can be easily understood by humans. As AI systems become more complex, understanding how they arrive at decisions becomes critical, especially in high-stakes domains like healthcare, finance, and criminal justice.",
        chunk_index=8,
        metadata={"topic": "explainable_ai", "category": "interpretability"}
    ),
    
    Chunk.from_text(
        text="Edge AI refers to artificial intelligence algorithms processed locally on hardware devices, rather than in the cloud. This approach reduces latency, improves privacy, and enables real-time processing for applications like autonomous vehicles, smart cameras, and IoT devices.",
        chunk_index=9,
        metadata={"topic": "edge_ai", "category": "deployment"}
    )
]

PHYSICS_CHUNKS: List[Chunk] = [
    Chunk.from_text(
        text="Classical Mechanics describes the motion of macroscopic objects, from projectiles to parts of machinery, and astronomical objects like spacecraft, planets, stars, and galaxies. It is based on Newton's three laws of motion and provides the foundation for understanding how forces affect the motion of physical objects.",
        chunk_index=0,
        metadata={"topic": "classical_mechanics", "category": "mechanics"}
    ),
    
    Chunk.from_text(
        text="Quantum Mechanics is the branch of physics that deals with the behavior of matter and light on the atomic and subatomic scale. It introduced concepts like wave-particle duality, uncertainty principle, and quantum entanglement, fundamentally changing our understanding of nature at the smallest scales.",
        chunk_index=1,
        metadata={"topic": "quantum_mechanics", "category": "modern_physics"}
    ),
    
    Chunk.from_text(
        text="Thermodynamics studies the relationships between heat, work, temperature, and energy. The four laws of thermodynamics govern the principles of energy transfer and describe how thermal energy is converted to and from other forms of energy and how it affects matter.",
        chunk_index=2,
        metadata={"topic": "thermodynamics", "category": "thermal_physics"}
    ),
    
    Chunk.from_text(
        text="Electromagnetism is one of the four fundamental forces of nature, describing the interaction between electrically charged particles. It encompasses electricity, magnetism, and light, unified by Maxwell's equations which describe how electric and magnetic fields propagate and interact with matter.",
        chunk_index=3,
        metadata={"topic": "electromagnetism", "category": "fundamental_forces"}
    ),
    
    Chunk.from_text(
        text="General Relativity, Einstein's theory of gravitation, describes gravity not as a force but as a consequence of the curvature of spacetime caused by mass and energy. It predicts phenomena like black holes, gravitational waves, and the expansion of the universe.",
        chunk_index=4,
        metadata={"topic": "general_relativity", "category": "relativity"}
    ),
    
    Chunk.from_text(
        text="Special Relativity introduces the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light is constant regardless of the observer's motion. This leads to time dilation, length contraction, and the famous equation E=mc².",
        chunk_index=5,
        metadata={"topic": "special_relativity", "category": "relativity"}
    ),
    
    Chunk.from_text(
        text="Particle Physics explores the fundamental constituents of matter and radiation, and their interactions. The Standard Model describes three of the four known fundamental forces and classifies all known elementary particles including quarks, leptons, and bosons like the Higgs boson.",
        chunk_index=6,
        metadata={"topic": "particle_physics", "category": "modern_physics"}
    ),
    
    Chunk.from_text(
        text="Optics is the study of light and its interactions with matter. It encompasses phenomena such as reflection, refraction, diffraction, and interference. Modern optics includes laser physics, fiber optics, and quantum optics, with applications in telecommunications, medicine, and computing.",
        chunk_index=7,
        metadata={"topic": "optics", "category": "waves"}
    ),
    
    Chunk.from_text(
        text="Condensed Matter Physics studies the physical properties of solid and liquid states of matter. It investigates phenomena like superconductivity, magnetism, crystalline structures, and phase transitions. This field has led to technological advances including semiconductors, transistors, and magnetic storage.",
        chunk_index=8,
        metadata={"topic": "condensed_matter", "category": "materials_science"}
    ),
    
    Chunk.from_text(
        text="Astrophysics applies the principles of physics to understand the universe beyond Earth. It studies celestial objects, cosmic phenomena, and the large-scale structure of the universe. Topics include stellar evolution, cosmology, dark matter, dark energy, and the Big Bang theory.",
        chunk_index=9,
        metadata={"topic": "astrophysics", "category": "astronomy"}
    )
]

DAD_JOKES_CHUNKS: List[Chunk] = [
    Chunk.from_text(
        text="Why don't scientists trust atoms? Because they make up everything!",
        chunk_index=0,
        metadata={"topic": "science_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="I'm reading a book about anti-gravity. It's impossible to put down!",
        chunk_index=1,
        metadata={"topic": "physics_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="Why did the scarecrow win an award? Because he was outstanding in his field!",
        chunk_index=2,
        metadata={"topic": "farm_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="I used to hate facial hair, but then it grew on me.",
        chunk_index=3,
        metadata={"topic": "grooming_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="What do you call a fake noodle? An impasta!",
        chunk_index=4,
        metadata={"topic": "food_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="Why don't eggs tell jokes? They'd crack each other up!",
        chunk_index=5,
        metadata={"topic": "food_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="I'm afraid for the calendar. Its days are numbered.",
        chunk_index=6,
        metadata={"topic": "time_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="Why did the math book look so sad? Because it had too many problems.",
        chunk_index=7,
        metadata={"topic": "math_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="What do you call a bear with no teeth? A gummy bear!",
        chunk_index=8,
        metadata={"topic": "animal_joke", "category": "dad_jokes"}
    ),
    
    Chunk.from_text(
        text="I told my wife she was drawing her eyebrows too high. She looked surprised.",
        chunk_index=9,
        metadata={"topic": "makeup_joke", "category": "dad_jokes"}
    )
]